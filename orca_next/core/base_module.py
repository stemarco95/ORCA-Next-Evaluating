from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from utils.context import Context


class BaseModule(ABC):
    """
    Abstract base class for all ORCA-Next modules.

    A module is a pure data-processing unit:
    - It consumes Observations from its input topics
    - It produces Observations on its output topics
    - It has no knowledge of scheduling, timing, or other modules
    """

    def __init__(
        self,
        module_id: str,
        inputs: List[str],
        outputs: List[str],
        cycle: int,
        seed: int,
        is_env: bool = False,
        config: Optional[dict] = None
    ):
        # --- Basic validation ---
        if not module_id:
            raise ValueError("module_id must be a non-empty string")

        if not isinstance(inputs, list):
            raise TypeError("inputs must be a list of topic names")

        if not isinstance(outputs, list):
            raise TypeError("outputs must be a list of topic names")

        if not isinstance(cycle, (float, int)) or cycle <= 0:
            raise ValueError("cycle must be a positive float or int")

        self.module_id: str = module_id
        self.inputs: List[str] = inputs
        self.outputs: List[str] = outputs
        self.cycle: int = cycle
        self.is_env: bool = is_env
        self.local_seed: int = seed
        # Scheduler-managed state
        self.last_execution = -float("inf")

        # Optional configuration
        self.config: dict = config or {}

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self, inputs: Dict[str, Context]) -> Dict[str, Context]:
        """
        Execute one logical step of the module.

        Args:
            inputs (Dict[str, Context]):
                Mapping from input topic to Observation.
                Guaranteed to contain all topics listed in self.inputs.

        Returns:
            Dict[str, Observation]:
                Mapping from output topic to Observation.
                Keys must be a subset of self.outputs.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional environment hook
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Context]:
        """
        Reset internal state of the module.

        Only meaningful for environment modules (is_env=True).
        Default implementation does nothing.

        Returns:
            Dict[str, Context]:
                Initial observations to publish (topic -> Observation).
        """
        return {}