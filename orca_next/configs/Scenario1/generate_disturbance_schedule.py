import json
import numpy as np

TARGET_PARAMETER = "other_speed"  # The environment parameter we want to disturb
RANGE = (4.0, 12.0)
PERIDOD_RANGE = (7000, 8000)  # Disturbance every x steps, where x is sampled from this range
FIRST_DISTURBANCE_STEP = 2020  # The step at which the first disturbance occurs

MAX_STEPS = 80000

INITIAL_VALUE = 7.0  # The initial value of the target parameter before any disturbances occur

if __name__ == "__main__":

    disturbances = []
    
    current_step = FIRST_DISTURBANCE_STEP

    last_value = INITIAL_VALUE
    while current_step < MAX_STEPS:
        value = round(np.random.uniform(*RANGE), 2)
        # Ensure the disturbance is significant enough
        while abs(value - last_value) < 3.0:
            value = round(np.random.uniform(*RANGE), 2)
        disturbances.append({
            "step": current_step,
            "target_parameter": TARGET_PARAMETER,
            "value": value
        })
        last_value = value
        period = np.random.randint(*PERIDOD_RANGE)
        current_step += period

    with open("configs/Scenario1/disturbance_schedule.json", "w") as f:
        json.dump({"disturbances": disturbances}, f)