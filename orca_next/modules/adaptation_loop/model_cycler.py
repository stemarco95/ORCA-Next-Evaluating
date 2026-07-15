from math import ceil, log2
import random
from typing import Dict

from core.audit_logger import AuditLogger
from core.base_module import BaseModule
from utils.context import Context

UTILITY_KEY = "utility" 
OUTPUT_KEY = "model_selection"

class ModelCycler(BaseModule):
    """Selects the active model by tracking utility and cycling when needed."""

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

        self.models = self.config.get("models", [])
        if not self.models:
            raise ValueError("The available models must be provided in the config under the 'models' key.")
        
        initial_model = self.config.get("initial_model")
        if initial_model and initial_model in self.models:
            self.idx = self.models.index(initial_model)
        else:            
            self.idx = 0  # Start with the first model by default
        
        self.rng = random.Random(self.local_seed)  # Create a local RNG for deterministic shuffling
        # Randomize the order of models to cycle through, but in a deterministic way
        # self.rng.shuffle(self.models)  # Randomize the order of models to cycle through
            
        self.acceptance_threshold = self.config.get("acceptance_threshold", 0.4)  # Default threshold for switching models based on utility


    def step(self, inputs: Dict[str, Context]) -> Dict[str, Context]:
        utility_ctx = inputs.get(UTILITY_KEY)
        utility = utility_ctx.info.get("utility")

        message_size = 0.0  # Default to zero unless we do a model switch

        enable_learning = False

        if utility is not None:
            if utility < self.acceptance_threshold:
                # Move to the next candidate when the current model underperforms.
                self.idx = (self.idx + 1) % len(self.models)
                enable_learning = False  # Disable learning when we switch models
                AuditLogger.log_event(f"Utility is low ({utility:.3f}), switching to model: {self.models[self.idx]}")

                message_size = ceil(log2(len(self.models)))  # Log the size of the model key message (in bits)
            else:
                # Keep the current model and allow learning when utility is acceptable.
                enable_learning = True # Enable learning when utility is above threshold
                AuditLogger.log_event(f"Utility is sufficient ({utility:.3f}), keeping current model: {self.models[self.idx]}")

        
        new_ctx = Context.with_info(info={
            "key": self.models[self.idx],
            "enable_learning": enable_learning,
            "message_size": message_size
        })
        return {OUTPUT_KEY: new_ctx}