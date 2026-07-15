import json
import numpy as np

TARGET_PARAMETER = "target_distance"  # The environment parameter we want to disturb
RANGE = (4.0, 20.0)
PERIDOD_RANGE = (14000, 16000)  # Disturbance every x steps, where x is sampled from this range
FIRST_DISTURBANCE_STEP = 1000  # The step at which the first disturbance occurs

MAX_STEPS = 250000

INITIAL_VALUE = 10.0  # The initial value of the target parameter before any disturbances occur

if __name__ == "__main__":

    disturbances = []
    
    current_step = FIRST_DISTURBANCE_STEP

    last_value = INITIAL_VALUE
    while current_step < MAX_STEPS:
        value = round(np.random.uniform(*RANGE), 2)
        # Ensure the disturbance is significant enough
        while abs(value - last_value) < 3.0 or abs(value - last_value) > 5.0:
            value = round(np.random.uniform(*RANGE), 2)
        disturbances.append({
            "step": current_step,
            "target_parameter": TARGET_PARAMETER,
            "value": value
        })
        last_value = value
        period = np.random.randint(*PERIDOD_RANGE)
        current_step += period

    with open("configs/Scenario3/disturbance_schedule.json", "w") as f:
        json.dump({"disturbances": disturbances}, f)