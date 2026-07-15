from typing import Dict

import numpy as np
from stable_baselines3 import TD3
from core.base_module import BaseModule
from utils.context import Context

ENV_STATE_KEY = "observation"
UPDATE_KEY = "model_update"
MODEL_KEY = "model"

OUTPUT_CONTEXT = "raw_action"


class Actor(BaseModule):
    """Operational actor that generates actions from observations using a TD3 policy."""

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

        # initialize random number generator with the provided seed for reproducibility
        self.rng = np.random.default_rng(seed=self.local_seed)

    def step(self, inputs: Dict[str, Context]) -> Dict[str, Context]:
        """Compute the next action, optionally adding exploration noise during training."""
        env_state_ctx = inputs.get(ENV_STATE_KEY)
        observation: np.ndarray = env_state_ctx.state

        model_ctx = inputs.get(MODEL_KEY)
        model: TD3 = model_ctx.info.get("model")

        # If online training is enabled, adaptation provides training flags/noise parameters.
        is_training = False
        if UPDATE_KEY in inputs:
            update_ctx = inputs.get(UPDATE_KEY)
            is_training = update_ctx.info.get("is_training")
            noise_std = update_ctx.info.get("noise_std")
             
        action: np.ndarray = model.predict(observation)[0]  # TD3's predict returns a tuple of (action, state), we only need the action here

        if is_training:
            noise = self.rng.normal(0, noise_std, size=action.shape)
            action = action + noise
            action = np.clip(action, -1.0, 1.0)
            
        return {
            OUTPUT_CONTEXT: Context.with_info(
                {
                    "action": action
                }
            )
        }