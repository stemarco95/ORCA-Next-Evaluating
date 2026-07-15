from typing import Any, Dict, List

from monitor.base_metric import BaseMetric

class Normalization(BaseMetric):
    """
    Subclass for Feature Scaling (Min-Max Normalization).
    Scales an input value to a [0, 1] range based on configured bounds.
    """

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

    def _validate_config(self, config: dict):
        """
        Ensure bounds are provided and min < max.
        """
        if "min" not in config or "max" not in config:
            raise ValueError(f"Normalization '{self.name}' requires 'min' and 'max' values.")
        
        if config["min"] >= config["max"]:
            raise ValueError(f"Normalization '{self.name}': 'min' must be less than 'max'.")
        
        if len(config.get("inputs", [])) != 1:
            raise ValueError(f"Normalization '{self.name}' expects exactly one input.")

    def _setup_operation(self, config: dict):
        """
        Store the bounds and the target input key.
        """
        self.clip = config.get("clip", True)  # Optional: whether to clip out-of-bounds values to [0, 1]
        self.min_val = config["min"]
        self.max_val = config["max"]
        self.range = self.max_val - self.min_val
        self.key = self.inputs[0]  # The key to extract from the input context for the normalization calculation

    def _apply_operation(self) -> Any:
        """
        Calculates: (value - min) / (max - min)
        """
        val = self.get_last_input.get(self.key)
        
        if val is None:
            return None

        # Perform min-max scaling
        normalized_val = (val - self.min_val) / self.range
        
        # Optional: Clip the result to [0, 1] to handle out-of-bounds values
        if self.clip:
            normalized_val = max(0.0, min(1.0, normalized_val))
        return normalized_val