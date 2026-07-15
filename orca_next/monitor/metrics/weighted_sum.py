from typing import Any, Dict, List

from monitor.base_metric import BaseMetric

class WeightedSum(BaseMetric):
    """
    Subclass for Weighted Sum Aggregation.
    Combines multiple state variables or metrics into a single value.
    """
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

    def _validate_config(self, config: dict):
        """
        Ensure every input has a corresponding weight.
        """
        inputs = config.get("inputs", [])
        if not inputs:
            raise ValueError(f"WeightedSum '{self.name}' requires at least one input.")
        
        for inp in inputs:
            if "weight" not in inp:
                raise ValueError(f"WeightedSum '{self.name}': Input '{inp.get('name')}' is missing a 'weight'.")

    def _setup_operation(self, config: dict):
        """
        Create a mapping of input names to their weights.
        """
        # We store weights in a dict for fast lookup during apply_operation
        self.weights = {
            inp["name"]: float(inp["weight"]) 
            for inp in config["inputs"]
        }

    def _apply_operation(self) -> Any:
        """
        Calculates: sum(value_i * weight_i)
        """
        total_sum = 0.0
        
        for name, weight in self.weights.items():
            val = self.get_last_input.get(name)
            
            if val is not None:
                try:
                    total_sum += float(val) * weight
                except (ValueError, TypeError):
                    # Handle cases where the input might not be a number (e.g. None or string)
                    raise ValueError(f"WeightedSum '{self.name}': Input '{name}' has non-numeric value '{val}'.")
        
        return total_sum