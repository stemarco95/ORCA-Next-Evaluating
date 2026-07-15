import json
import numpy as np

TARGET_PARAMETER = "soft_error"  # The environment parameter we want to disturb
RANGE = (1e-1, 2e-1)  # The range from which to sample disturbance values for the target parameter
PERIDOD_RANGE = (9000, 11000)  # Disturbance every x steps, where x is sampled from this range
FIRST_DISTURBANCE_STEP = 1500  # The step at which the first disturbance occurs

MAX_STEPS = 110000

INITIAL_VALUE = 0.0  # The initial value of the target parameter before any disturbances occur

if __name__ == "__main__":

    disturbances = []
    
    current_step = FIRST_DISTURBANCE_STEP

    last_value = INITIAL_VALUE
    while current_step < MAX_STEPS:
        value = round(np.random.uniform(*RANGE), 5)
        # Ensure the disturbance is significant enough
        
        disturbances.append({
            "step": current_step,
            "target_parameter": TARGET_PARAMETER,
            "value": value
        })
        last_value = value
        period = np.random.randint(*PERIDOD_RANGE)
        current_step += period

    with open("configs/Scenario4/disturbance_schedule.json", "w") as f:
        json.dump({"disturbances": disturbances}, f)