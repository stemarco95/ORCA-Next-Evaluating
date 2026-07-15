from typing import Any, Dict, List

from monitor.base_metric import BaseMetric

class Exponential(BaseMetric):
    """
    Metric that calculates the exponential of a numerical input.
    """
    def __init__(self, name : str, config: dict):
        super().__init__(name, config)

    def _validate_config(self, config: dict):
        if len(config.get("inputs", [])) != 1:
            raise ValueError(f"Exponential '{self.name}' expects exactly one input.")
        
        if config.get("base") is None:
            raise ValueError(f"Exponential '{self.name}' requires a 'base' in the configuration.")

        
    def _setup_operation(self, config: dict):
        self.base = config["base"]
        self.key = self.inputs[0]  # The key to extract from the input context for the exponential calculation

    def _apply_operation(self) -> Any:
        val = self.get_last_input.get(self.key)

        if val is None:
            return None 
        
        return self.base ** val
