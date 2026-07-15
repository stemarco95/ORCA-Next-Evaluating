from typing import Dict

from math import ceil, log2

from core.base_module import BaseModule
from core.audit_logger import AuditLogger
from utils.context import Context

UTILITY_KEY = "utility" 
OUTPUT_KEY = "model_selection"
DISTURBANCE_KEY = "disturbance"

class PerfectSelector(BaseModule):
    """Selects a model key from rule mappings when utility drops below a threshold."""

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

        rules = self.config.get("rules", [])
        if not rules:
            raise ValueError("The rules for model selection must be provided in the config under the 'rules' key.")
        
        self.models = {}
        for rule in rules:
            # expect a dictionary with keys: "target_parameter", "value", "model_key"
            if not all(k in rule for k in ("target_parameter", "value", "model_key")):
                raise ValueError("Each rule must contain 'target_parameter', 'value', and 'model_key' keys.")
            
            target_parameter = rule["target_parameter"]
            value = rule["value"]
            model_key = rule["model_key"]   

            self.models[(target_parameter, value)] = model_key

        initial_model = self.config.get("initial_model")
        if initial_model and initial_model in self.models.values():
            self.model_key = initial_model
        else:
            # Set a default model key, the first one in the rules
            self.model_key = rules[0]["model_key"]
        
        self.acceptance_threshold = self.config.get("acceptance_threshold", 0.4)  # Default threshold for switching models based on utility

        self.find_closest = self.config.get("find_closest", False)  # Whether to find the closest matching rule if no exact match is found


        self.idx = 0

    def step(self, inputs: Dict[str, Context]) -> Dict[str, Context]:
        """Return the currently selected model and communication cost for this step."""

        utility_ctx = inputs.get(UTILITY_KEY)
        utility = utility_ctx.info.get("utility")
        disturbance_ctx = inputs.get(DISTURBANCE_KEY)
        target_parameter = disturbance_ctx.info.get("target_parameter") if disturbance_ctx else None
        value = disturbance_ctx.info.get("value") if disturbance_ctx else None
    
        message_size = 0.0  # Default to zero unless we do a model switch

        if utility is not None:
            if utility < self.acceptance_threshold:
                # Find the model that matches the current disturbance condition
                self.model_key = self.models.get((target_parameter, value))
                if self.model_key:
                    AuditLogger.log_event(f"Utility is low ({utility:.3f}) and disturbance condition ({target_parameter}={value}) matches rule, switching to model: {self.model_key}")
                    message_size = ceil(log2(len(self.models)))  # Log the size of the model key message (in bits)
                else:
                    if self.find_closest:
                        # Find the closest matching rule
                        closest_key = min(self.models.keys(), key=lambda k: abs(k[1] - value) if k[0] == target_parameter else float('inf'))
                        self.model_key = self.models[closest_key]
                        message_size = ceil(log2(len(self.models)))  # Log the size of the model key message (in bits)
                        AuditLogger.log_event(f"Utility is low ({utility:.3f}) but no exact matching disturbance condition for ({target_parameter}={value}), switching to closest model: {self.model_key}")
                    else:
                        AuditLogger.log_event(f"Utility is low ({utility:.3f}) but no matching disturbance condition for ({target_parameter}={value}), keeping current model: {self.model_key}")
            else:
                AuditLogger.log_event(f"Utility is sufficient ({utility:.3f}), keeping current model: {self.model_key}")

        
        new_ctx = Context.with_info(info={
            "key": self.model_key,
            "message_size": message_size
        })
        return {OUTPUT_KEY: new_ctx}