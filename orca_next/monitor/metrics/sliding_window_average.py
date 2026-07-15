from collections import deque
from typing import Any, Dict, List

from monitor.base_metric import BaseMetric

class SlidingWindowAverage(BaseMetric):
    """
    Subclass for Sliding Window Average (SMA).
    Calculates the average of a specific input over a sliding window.
    """
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

    def _validate_config(self, config: dict):
        """
        Ensure the config has a window_size and exactly one input.
        """
        if "window_size" not in config:
            raise ValueError(f"SlidingWindowAverage Metric '{self.name}' requires 'window_size' in config.")
        
        inputs = config.get("inputs", [])
        if len(inputs) != 1:
            # SlidingWindowAverage typically operates on a single stream of data
            raise ValueError(f"SlidingWindowAverage Metric '{self.name}' expects exactly one input, got {len(inputs)}.")

    def _setup_operation(self, config: dict):
        """
        Initialize the sliding window buffer.
        """
        self.window_size = config["window_size"]
        # Deque with maxlen automatically handles the sliding window (circular buffer)
        self.buffer = deque(maxlen=self.window_size)
        self.key = self.inputs[0]  # The key to extract from the input context for the SMA calculation

    def _apply_operation(self) -> Any:
        """
        Applies the SMA logic. 
        """
        history_length = len(self.get_history) # should be equal to update_cycle

        # for performance reasons, we can avoid storing all history values in the buffer if the window size is smaller than the history length
        if self.window_size <= history_length:
            start_index = history_length - self.window_size
            value = sum([self.get_history[i].get(self.key) for i in range(start_index, history_length)]) / self.window_size
        else:
            values = [self.get_history[i].get(self.key) for i in range(history_length)]
            self.buffer.extend(values)
            value = sum(self.buffer) / self.window_size

    
        return value