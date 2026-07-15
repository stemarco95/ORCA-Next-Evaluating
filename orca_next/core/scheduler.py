import time
from typing import List, Dict
from collections import deque

from concurrent.futures import ThreadPoolExecutor

from core.messages import Message
from core.mediator import Mediator
from core.base_module import BaseModule
from core.audit_logger import AuditLogger
from utils.context import Context


class Scheduler:
    """
    Central scheduler responsible for deciding
    - which modules are due
    - which modules are executable (inputs available)
    - how execution is performed (inline vs threaded)
    """

    def __init__(
        self,
        modules: List[BaseModule],
        mediator: Mediator,
        mode: str,
        max_workers: int = 4,
    ):
        self.modules = modules
        self.mediator = mediator
        self.mode = mode

        self.min_cycle = min((m.cycle for m in self.modules), default=100)

        self.executor = (
            ThreadPoolExecutor(max_workers=max_workers)
            if mode == "time-thread-based"
            else None
        )

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _can_execute(self, module: BaseModule) -> bool:
        return all(topic in self.mediator.latest_messages for topic in module.inputs)

    def _collect_inputs(self, module: BaseModule) -> Dict[str, Context]:
        inputs: Dict[str, Context] = {}

        for topic in module.inputs:
            msg = self.mediator.latest_messages.get(topic)
            if msg is None:
                continue

            if not isinstance(msg.payload, Context):
                raise TypeError(
                    f"Module '{module.module_id}' expected Context on topic '{topic}', "
                    f"got {type(msg.payload)}"
                )

            inputs[topic] = msg.payload

        return inputs

    # ------------------------------------------------------------------
    # Module execution
    # ------------------------------------------------------------------

    def _run_module(self, module: BaseModule):
        try:
            if not self._can_execute(module):
                return

            inputs = self._collect_inputs(module)
            outputs = module.step(inputs) or {}

            if not isinstance(outputs, dict):
                raise TypeError(
                    f"Module '{module.module_id}' must return dict(topic->Observation)"
                )

            for topic, obs in outputs.items():
                if not isinstance(obs, Context):
                    raise TypeError(
                        f"Output '{topic}' of module '{module.module_id}' "
                        f"must be Observation, got {type(obs)}"
                    )

                self.mediator.publish(
                    Message(topic=topic, payload=obs, sender=module.module_id),
                    cycle=module.cycle,
                    persist_reset=module.is_env
                )

            AuditLogger.log_module_execution(module_id=module.module_id)

            module.last_execution = time.time()

        except Exception as e:
            AuditLogger.log_event(
                "execution_error",
                module=module.module_id,
                error=str(e),
            )

    def _execute(self, module: BaseModule):
        if self.executor:
            self.executor.submit(self._run_module, module)
        else:
            self._run_module(module)

    # ------------------------------------------------------------------
    # Step-based mode
    # ------------------------------------------------------------------

    def run_step(self, step: int):
        runnable = deque(
            m for m in self.modules
            if step % m.cycle == 0
        )

        no_progress = 0

        # Clear modules list and re-populate with executed modules to maintain order
        if step == 0:
            self.modules = [] 

        while runnable:
            module = runnable.popleft()

            if self._can_execute(module):
                self._execute(module)
                if step == 0:
                    self.modules.append(module)  # Add back to main list after execution
                no_progress = 0
            else:
                runnable.append(module)
                no_progress += 1

            if no_progress > len(runnable):
                AuditLogger.log_event(
                    "deadlock_detected",
                    mode="step-based",
                    step=step,
                    modules=[m.module_id for m in runnable],
                )
                break

    # ------------------------------------------------------------------
    # Time-based mode
    # ------------------------------------------------------------------

    def run_time_based(self):
        """
        Executes one scheduling cycle.
        """

        runnable = deque(
            m for m in self.modules
            if (time.time() - m.last_execution) >= m.cycle
        )

        no_progress = 0

        while runnable:
            module = runnable.popleft()

            if self._can_execute(module):
                self._execute(module)
                no_progress = 0
            else:
                runnable.append(module)
                no_progress += 1

            if no_progress > len(runnable):
                AuditLogger.log_event(
                    "deadlock_detected",
                    mode="time-based",
                    modules=[m.module_id for m in runnable],
                )
                break
