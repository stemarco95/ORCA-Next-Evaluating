from typing import Any, Dict, List

from monitor.base_metric import BaseMetric

class Constant(BaseMetric):
    """
    Metric that returns a constant value.
    """
    def __init__(self, name : str, config: dict):
        super().__init__(name, config)

    def _validate_config(self, config: dict):
        if len(config.get("inputs", [])) != 0:
            raise ValueError(f"Constant '{self.name}' expects exactly zero inputs.")
        
        if "value" not in config:
            raise ValueError(f"Constant '{self.name}' requires a 'value' in the configuration.")
        
        if not isinstance(config["value"], (int, float)):
            raise ValueError(f"Constant '{self.name}' expects 'value' to be a number (int or float).")

    def _setup_operation(self, config: dict):
        self.value = config["value"]

    def _apply_operation(self) -> Any:
        return self.value