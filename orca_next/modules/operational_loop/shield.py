"""Shield module: simple kinematic safety check for longitudinal control.

This module inspects a raw normalized action and the current environment
observation (distance and relative speed) and optionally overrides the
action with a safe braking command when a short-horizon collision risk
is detected.

The implementation is intentionally lightweight and intended for use as
an operational safety layer (a "shield").
"""

from typing import Dict

import numpy as np
from core.base_module import BaseModule
from utils.context import Context

ENV_STATE_KEY = "observation"
ACTION_KEY = "raw_action"

OUTPUT_CONTEXT = "safe_action"

class Shield(BaseModule):
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

        # Toggle to quickly enable/disable the shield without changing
        # upstream controllers.
        self.enabled = self.config.get("enabled", True)

    def step(self, inputs: Dict[str, Context]) -> Dict[str, Context]:
        """Process one step: return a context containing a possibly
        modified (safe) action and metadata about the intervention.
        """

        action_ctx = inputs.get(ACTION_KEY)
        assert action_ctx is not None, f"Expected action context with key '{ACTION_KEY}' not found in inputs"
        action: np.ndarray = action_ctx.info.get("action")

        env_state_ctx = inputs.get(ENV_STATE_KEY)
        assert env_state_ctx is not None, f"Expected environment state context with key '{ENV_STATE_KEY}' not found in inputs"
        
        distance = env_state_ctx.info.get("distance")
        # Assuming positive relative_speed means the ego vehicle is closing the gap
        relative_speed = env_state_ctx.info.get("relative_speed") 

        # --- KINEMATIC SAFETY CHECK ---
        # 1. Map the normalized action [-1.0, 1.0] to physical acceleration [-5.0, 5.0] m/s^2
        max_accel = 5.0
        ego_accel = float(action[0]) * max_accel
        
        # 2. Safety lookahead horizon (seconds). Small horizon to keep the
        # check conservative and computationally cheap.
        t_horizon = 1.0
        
        # 3. Handle braking edge case: when braking, relative speed may reach
        # zero before the full horizon. Only simulate until closing stops to
        # avoid overestimating progress.
        if ego_accel < 0 and relative_speed > 0:
            time_to_stop_closing = -relative_speed / ego_accel
            t_eval = min(t_horizon, time_to_stop_closing)
        else:
            t_eval = t_horizon
            
        # 4. Project the future distance using: d(t) = d_0 - (v_0*t + 0.5*a*t^2)
        expected_distance = distance - (relative_speed * t_eval + 0.5 * ego_accel * (t_eval ** 2))

        # If the expected distance drops to a small safety margin within the
        # horizon, intervene by issuing a strong braking command.
        if expected_distance <= 1.0 and self.enabled:
            # Note: the control space upstream is normalized; this module
            # emits a physical-style braking value for simplicity.
            safe_action = np.array([-2.0])  # strong brake command
            shield_intervention = True
        else:
            safe_action = action  # No modification to the original action
            shield_intervention = False

        return {
            OUTPUT_CONTEXT: Context.with_info(
                {
                    "action": safe_action, 
                    "shield_intervention": shield_intervention,
                    "raw_action": action
                }
            )
        }