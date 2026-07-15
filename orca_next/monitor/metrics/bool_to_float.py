from typing import Any, Dict, List

from monitor.base_metric import BaseMetric

class BoolToFloat(BaseMetric):
    """
    Metric that converts a boolean input to a float (1.0 for True, 0.0 for False).
    Useful for metrics that need to be aggregated or combined with other numerical metrics.
    """
    def __init__(self, name : str, config: dict):
        super().__init__(name, config)

    def _validate_config(self, config: dict):
        if len(config.get("inputs", [])) != 1:
            raise ValueError(f"BoolToFloat '{self.name}' expects exactly one input.")
        
        input_desc = config["inputs"][0]
        if input_desc.get("type") != "state":
            raise ValueError(f"BoolToFloat '{self.name}' expects an input of type 'state'.")

    def _setup_operation(self, config: dict):
        self.key = self.inputs[0]  # The key to extract from the input context for the boolean value

    def _apply_operation(self) -> Any:
        val = self.get_last_input.get(self.key)
        
        if val is None:
            return None
        
        if self.config.get("invert", False):
            return 0.0 if val else 1.0
        else:
            return 1.0 if val else 0.0