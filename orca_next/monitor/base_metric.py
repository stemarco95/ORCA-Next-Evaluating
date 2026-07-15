import inspect
import importlib
import os
from typing import Any, Dict, List
from collections import deque

from core.audit_logger import AuditLogger

def snake_to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase"""
    return "".join(part.capitalize() for part in name.split("_"))


def auto_load_metrics(directory: str = "monitor/metrics") -> dict:
    """
    Dynamically import all module classes from the given directory and subdirectories.
    Assumes:
        file name: snake_case
        class name: PascalCase version of it.
    
    Returns:
        dict: Mapping type_name -> class object
    """
    registry = {}

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py") and not filename.startswith("_"):
                mod_name = filename[:-3]
                class_name = snake_to_pascal(mod_name)
                
                # Build module path from directory structure
                rel_path = os.path.relpath(root, ".")
                module_path = rel_path.replace(os.sep, ".") + "." + mod_name

                try:
                    module = importlib.import_module(module_path)
                    cls = getattr(module, class_name, None)

                    if inspect.isclass(cls) and issubclass(cls, BaseMetric):
                        registry[class_name] = cls
                        AuditLogger.log_message(f"Metric loaded: {class_name} from {module_path}")
                    else:
                        AuditLogger.log_event(
                            "module_warning",
                            msg=f"Expected class '{class_name}' not found in '{module_path}'"
                        )

                except Exception as e:
                    AuditLogger.log_event(
                        "module_error",
                        module=module_path,
                        error=str(e)
                    )

    return registry

class BaseMetric:
    """
    Base class for all metrics and aggregators.
    """
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config


        self.save = config.get("save", None)
        self.plot = config.get("plot", False)
        self.publish = config.get("publish", False)
        self.update_cycle = config.get("update_cycle", 1)  # Optional: how often to update the metric (default every step)
        self.intial_value = config.get("initial_value", None)  # Optional: initial value for the metric
        self.last_value = None  # To store the last calculated value for update cycles

        if self.plot and self.save is None:
            self.save = True  # Default to saving if plotting is enabled
        elif self.save is None:
            self.save = False  # Default to not saving if not specified and not plotting
        elif self.plot and self.save is False:
            raise ValueError(f"Metric '{self.name}' cannot be plotted if it is not saved. Please set 'save' to true for this metric.")

        self._validate_config(config)

        self.__setup_inputs(config)
        self._setup_operation(config)

        self.history = []
            
    def _validate_config(self, config: dict):
        raise NotImplementedError("Config validation not implemented in BaseMetric. Please implement in subclass if needed.")

    def __setup_inputs(self, config: dict):
        input_desc = config.get("inputs", []) 
        self.input_states = []
        self.input_metrics = []
        self.inputs = []  # For metrics that specify a key to extract from the input context
        for inp in input_desc:
            inp_type = inp.get("type")
            inp_name = inp.get("name")
            if inp_type is None or inp_name is None:
                raise ValueError(f"Invalid input description: {inp}")
            if inp_type not in ["state", "metric"]:
                raise ValueError(f"Invalid input description: {inp}")
            
            if inp_type == "state":
                self.input_states.append(inp_name)
            elif inp_type == "metric":
                self.input_metrics.append(inp_name)

            self.inputs.append(inp_name)  # Optional key for extracting specific value from context
   
    def _setup_operation(self, config: dict):
        # For simplicity, we will not implement aggregation in this base class, but it can be extended in subclasses
        raise NotImplementedError("Operation setup not implemented in BaseMetric. Please implement in subclass if needed.")

    def __call__(self, oc_state: List[Dict], computed_metrics: Dict[str, Any]) -> Any:
        """
        Calculate the metric value.
        :param oc_state: The current state from OcState
        :param computed_metrics: Dictionary of metrics already calculated in this step
        """
        inputs = self.__extract_inputs(oc_state, computed_metrics)
        step = computed_metrics.get("step", 1)   
        
        self.history.append(inputs)

        if self.intial_value is not None and self.last_value is None:
            self.last_value = self.intial_value
            return self.intial_value

        if step % self.update_cycle == 0:
            # Update the last calculated value and return it
            self.last_value = self._apply_operation()
            self.history.clear()  # Clear history after applying operation
   
        return self.last_value

    def __extract_inputs(self, oc_state: List[Dict], computed_metrics: Dict[str, Any]) -> Dict[str, Any]:
        inputs = {}
        for state_name in self.input_states:
            if state_name not in oc_state:
                raise ValueError(f"State '{state_name}' not found in OC state")
            inputs[state_name] = oc_state[state_name]
        
        for metric_name in self.input_metrics:
            if metric_name not in computed_metrics:
                raise ValueError(f"Metric '{metric_name}' not found among computed metrics")
            inputs[metric_name] = computed_metrics[metric_name]
        
        return inputs

    def _apply_operation(self) -> Any:
        # Base implementation does no operation, just returns the inputs as a dict
        raise NotImplementedError("Operation not implemented in BaseMetric. Please implement in subclass.")
    
    def reset(self):
        raise NotImplementedError("Reset not implemented in BaseMetric. Please implement in subclass if needed.")
   
    def inputs_available(self, oc_state: List[Dict], computed_metrics: Dict[str, Any]) -> bool:
        """
        Check if all required inputs for this metric are available.
        """
        for state_name in self.input_states:
            if state_name not in oc_state or oc_state[state_name] is None:
                return False
        
        for metric_name in self.input_metrics:
            if metric_name not in computed_metrics or computed_metrics[metric_name] is None:
                return False
        
        return True

    def uses_only_states(self) -> bool:
        """
        Check if the metric uses only state inputs.

        Returns:
            bool: True if the metric uses only state inputs, False otherwise.
        """
        return len(self.input_metrics) == 0
    
    def has_to_be_saved(self) -> bool:
        """
        Check if the metric is configured to be saved.

        Returns:
            bool: True if the metric should be saved, False otherwise.
        """
        return self.save
    
    def has_to_be_published(self) -> bool:
        """
        Check if the metric is configured to be published.

        Returns:
            bool: True if the metric should be published, False otherwise.
        """
        return self.publish
    
    @property
    def get_last_input(self) -> Dict[str, Any]:
        """
        Get the last input values for this metric.

        Returns:
            Dict[str, Any]: The last input values.
        """
        if self.history:
            return self.history[-1]
        return {}
    
    @property
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of input values for this metric.

        Returns:
            List[Dict[str, Any]]: The history of input values.
        """
        return self.history