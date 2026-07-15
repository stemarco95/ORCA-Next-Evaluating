from collections import deque
from typing import Any, Dict, List

from monitor.base_metric import BaseMetric

class ExponentialMovingAverage(BaseMetric):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

    def _validate_config(self, config: dict):
        """
        Ensure the config has a smoothing_factor and exactly one input.
        """
        if "smoothing_factor" not in config:
            raise ValueError(f"ExponentialMovingAverage Metric '{self.name}' requires 'smoothing_factor' in config.")
        
        inputs = config.get("inputs", [])
        if len(inputs) != 1:
            raise ValueError(f"ExponentialMovingAverage Metric '{self.name}' expects exactly one input, got {len(inputs)}.")
        
    def _setup_operation(self, config: dict):
        """
        Initialize the EMA state.
        """
        self.smoothing_factor = config["smoothing_factor"]
        self.ema_value = None  # Will hold the current EMA value
        self.key = self.inputs[0]  # The key to extract from the input context for the EMA calculation
 

    def _apply_operation(self) -> Any:
        """
        Applies the EMA logic. 
        """
        # 1. Get the current value from the extracted inputs
        current_val = self.get_last_input.get(self.key)
        
        if current_val is None:
            return None

        # 2. Update EMA value
        if self.ema_value is None:
            self.ema_value = current_val  # Initialize with the first value
        else:
            self.ema_value = (self.smoothing_factor * current_val) + ((1 - self.smoothing_factor) * self.ema_value)

        return self.ema_value
