from typing import Dict

import numpy as np
from core.base_module import BaseModule
from utils.context import Context

OUTPUT_CONTEXT = "raw_action"

class DummyActor(BaseModule):
    def __init__(
            self, 
            module_id, 
            inputs, 
            outputs, 
            cycle, 
            seed,
            is_env=False, 
            config=None
        ):
        super().__init__(module_id, inputs, outputs, cycle, seed, is_env, config)

    def step(self, inputs: Dict[str, Context]) -> Dict[str, Context]:
             
        action = np.array([1.0])  # Dummy action, replace with actual logic to compute action based on observation and model
            
        return {
            OUTPUT_CONTEXT: Context.with_info(
                {
                    "action": action
                }
            )
        }