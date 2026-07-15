import json
from typing import Dict
from core.base_module import BaseModule
from utils.context import Context

OUTPUT_CONTEXT = "disturbance"

DISTURBANCE_MODES = ["other_speed", "obs_x_offset", "target_distance", "soft_error"]

DISTURBANCE_RANGES = {
    "other_speed": (4.0, 14.0),  # Speed of other vehicles in m/s
    "obs_x_offset": (-7.0, 7.0),  # Lateral offset of obstacles in meters
    "target_distance": (4.0, 20.0),  # Desired distance to lead vehicle in meters
    "soft_error": (0.0, 1.0)  # Soft error rate as a probability
}

class DisturbanceGenerator(BaseModule):
    def __init__(
            self, 
            module_id, 
            inputs, 
            outputs, 
            cycle, 
            seed,
            is_env=False, 
            config=None
        ):
        super().__init__(module_id, inputs, outputs, cycle, seed, is_env, config)

        # Step counter used to trigger disturbance events at the configured time.
        self.step_count = 0
 
        self.target_parameter = self.config.get("target_parameter", None)
        assert self.target_parameter is not None, "target_parameter must be specified in the DisturbanceGenerator config"
        assert self.target_parameter in DISTURBANCE_MODES, "Unsupported target_parameter specified in DisturbanceGenerator config. Supported values are: " + ", ".join(DISTURBANCE_MODES)

        self.initial_value = self.config.get("initial_value", None)
        assert self.initial_value is not None, "initial_value must be specified in the DisturbanceGenerator config"
        min_val, max_val = DISTURBANCE_RANGES[self.target_parameter]
        assert min_val <= self.initial_value <= max_val, f"initial_value for {self.target_parameter} must be between {min_val} and {max_val}"
        self.current_value = self.initial_value  # Initialize current value to the initial value

        # Support both in-memory schedules and schedules loaded from JSON.
        self.disturbances = self.config.get("disturbances", None)
        assert self.disturbances is not None, "disturbances must be specified in the DisturbanceGenerator config"
        assert isinstance(self.disturbances, (list, str)), "disturbances must be a list of disturbance events or a string pointing to a JSON file containing the disturbance schedule"
        
        if isinstance(self.disturbances, str):
            # Load disturbance schedule from JSON file
            with open(self.disturbances, 'r') as f:
                d = json.load(f)
                self.disturbances = d.get("disturbances", None)
                assert self.disturbances is not None, "The disturbance schedule JSON file must contain a 'disturbances' key with a list of disturbance events"
                assert isinstance(self.disturbances, list), "The disturbance schedule JSON file must contain a list of disturbance events"

        last_step = 0
        for disturbance in self.disturbances:
            assert "step" in disturbance and "value" in disturbance, "Each disturbance event must have 'step' and 'value' keys"

            assert isinstance(disturbance["step"], int) and disturbance["step"] > 0, "'step' must be a positive integer"
            assert disturbance["step"] > last_step, "Disturbance events must be in increasing order of steps"
            last_step = disturbance["step"]

            assert isinstance(disturbance["value"], (int, float)), "'value' must be a number"
            min_val, max_val = DISTURBANCE_RANGES[self.target_parameter]
            assert min_val <= disturbance["value"] <= max_val, f"'value' for {self.target_parameter} must be between {min_val} and {max_val}"
            

    def step(self, inputs: Dict[str, Context]) -> Dict[str, Context]:
        """
        Advances the simulation by one step and applies the next scheduled disturbance when due.
        """
        self.step_count += 1

        if self.disturbances and self.disturbances[0]["step"] <= self.step_count:
            self.current_value = self.disturbances[0]["value"]
            self.disturbances.pop(0)  # Remove the disturbance event that has been applied
        elif self.target_parameter == "soft_error":
            self.current_value = 0.0  # Reset to no error when not disturbed

        return {OUTPUT_CONTEXT: Context.with_info(
            info={
                "target_parameter": self.target_parameter,
                "value": self.current_value
            })}
    

        """Example config for a disturbance schedule.

        .. code-block:: json

                {
                    "config": {
                        "disturbance_schedule": {
                            "target_parameter": "other_speed",
                            "initial_value": 8.0,
                            "disturbances": [
                                {"step": 100, "value": 12.0},
                                {"step": 200, "value": 6.0},
                                {"step": 300, "value": 10.0}
                            ]
                        }
                    }
                }
        """