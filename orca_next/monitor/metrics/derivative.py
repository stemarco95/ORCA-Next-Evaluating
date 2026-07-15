from typing import Any, Dict, List

from monitor.base_metric import BaseMetric

class Derivative(BaseMetric):
    """
    Metric that calculates the derivative of a numerical input.
    Useful for tracking rates of change, such as acceleration (derivative of speed) or jerk (derivative of acceleration)."""
    def __init__(self, name : str, config: dict):
        super().__init__(name, config)
        self.previous_value = None

    def _validate_config(self, config: dict):
        if len(config.get("inputs", [])) != 1:
            raise ValueError(f"Derivative '{self.name}' expects exactly one input.")
        
    def _setup_operation(self, config: dict):
        self.key = self.inputs[0]  # The key to extract from the input context for the derivative calculation

    def _apply_operation(self) -> Any:
        val = self.get_last_input.get(self.key)
        
        if val is None:
            return None
        
        if self.previous_value is None:
            derivative = 0.0  # No previous value, so we define derivative as 0
        else:
            derivative = val - self.previous_value

        self.previous_value = val
        return derivative