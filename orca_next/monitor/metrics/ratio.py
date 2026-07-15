from typing import Any, Dict, List

from monitor.base_metric import BaseMetric

class Ratio(BaseMetric):
    """
    Metric that calculates the ratio of two numerical inputs.
    Useful for tracking proportions or relative changes."""
    def __init__(self, name : str, config: dict):
        super().__init__(name, config)

    def _validate_config(self, config: dict):
        if len(config.get("inputs", [])) != 2:
            raise ValueError(f"Ratio '{self.name}' expects exactly two inputs.")
        
    def _setup_operation(self, config: dict):
        self.nominator_input_name = config["inputs"][0]["name"]
        self.denominator_input_name = config["inputs"][1]["name"]

    def _apply_operation(self) -> Any:
        numerator = self.get_last_input.get(self.nominator_input_name)
        denominator = self.get_last_input.get(self.denominator_input_name)

        if numerator is None or denominator is None:
            return None

        if denominator == 0:
            return float('inf')  # Define ratio as infinity if denominator is zero to avoid division by zero error

        return numerator / denominator