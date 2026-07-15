from typing import Any, Dict, List

from monitor.base_metric import BaseMetric

class Absolute(BaseMetric):
    """
    Metric that calculates the absolute value of a numerical input.
    Useful for tracking the magnitude of a quantity regardless of its sign."""
    def __init__(self, name : str, config: dict):
        super().__init__(name, config)

    def _validate_config(self, config: dict):
        if len(config.get("inputs", [])) != 1:
            raise ValueError(f"Absolute '{self.name}' expects exactly one input.")
        
    def _setup_operation(self, config: dict):
        self.key = self.inputs[0]  # The key to extract from the input context for the absolute value calculation

    def _apply_operation(self) -> Any:
        val = self.get_last_input.get(self.key)
        
        if val is None:
            return None
        
        return abs(val)