from time import sleep
from typing import Dict
import gymnasium as gym
import numpy as np

from core.base_module import BaseModule
from utils.context import Context
from utils.register_gym import *  # Ensure the custom environment is registered


ACTION_KEY = "safe_action"
DISTURBANCE_KEY = "disturbance"

class Environment(BaseModule):
    """Environment wrapper for stepping, resets, and experience bookkeeping."""

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

        self.render = self.config.get("render", False)
        self.render_mode = "human" if self.render else None
        
        self.randomize_speed = self.config.get("randomize_speed", False)
        self.other_speed = self.config.get("other_speed", 8.0)

        self.env_config = {
            "generalize": self.randomize_speed,
            "target_distance": 10.0,
            "other_speed": self.other_speed,
            "obs_x_offset": 0.0,
            "distance_noise": 1.0, 
            "initial_speed": [0.0, 2.0],
            "far_away_penalty": False,
            "generalize_speed_range": [6.0, 10.0]
        }
        
        self.env = gym.make(
            "acc-v0", 
            config=self.env_config, 
            render_mode=self.render_mode
        )

        # Reset env ONCE with seed to ensure deterministic initial state for the first episode.
        self.env.reset(seed=self.local_seed)

        self.last_observation = None
        self.gathering_experience = True
        self.safe_mode = False  # When set, the next action is replaced with a safe fallback.


    def _update_env_config(self, env_ctx: Context):
        """Update the active environment configuration from disturbance context values."""
        if env_ctx:
            target_parameter = env_ctx.info.get("target_parameter")
            value = env_ctx.info.get("value")
            if target_parameter and value is not None:
                if target_parameter == "other_speed" and self.randomize_speed:
                    raise ValueError("Received disturbance for other_speed, but randomize_speed is enabled. The disturbance will have no effect.")
                else:
                    self.env_config.update({target_parameter: value})

    def reset(self) -> Dict[str, Context]:
        # Re-apply the latest settings before starting a fresh episode.
        self.env.unwrapped.configure(self.env_config)  

        obs, info = self.env.reset()
      
        info["env_reset"] = True  # Signal that this observation came from a reset.

        self.last_observation = obs
        self.gathering_experience = True
        self.safe_mode = False
        info["experience"] = None  # Reset observations do not produce transitions.
        return {self.outputs[0]: Context(
            state=obs,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info
        )}
    
    def step(self, inputs: Dict[str, Context]) -> Dict[str, Context]:

        action_ctx = inputs.get(ACTION_KEY, None)
        assert action_ctx is not None, f"Expected action context with key '{ACTION_KEY}' not found in inputs"
        raw_action: np.ndarray = action_ctx.info.get("raw_action")
        action: np.ndarray = action_ctx.info.get("action")
        shield_intervention: bool = action_ctx.info.get("shield_intervention")

        disturbance_ctx = inputs.get(DISTURBANCE_KEY, None)
        assert disturbance_ctx is not None, f"Expected disturbance context with key '{DISTURBANCE_KEY}' not found in inputs"
        self._update_env_config(disturbance_ctx)

        if self.safe_mode:
            action = np.array([0.0])  # Use a neutral action while safe mode is active.

        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.render:
            sleep(0.02)  # Sleep to slow down rendering for human viewing

        info["env_reset"] = False  # Mark this as a regular step.
        info["shield_intervention"] = shield_intervention

        # Persist the transition only while experience gathering remains enabled.
        if self.last_observation is not None and self.gathering_experience:
            learning_episode_done = terminated or truncated or shield_intervention
            info["experience"] = (self.last_observation, obs, raw_action, reward, learning_episode_done, info)
        else:
            info["experience"] = None

        # Stop gathering experience after a shield intervention until the next reset.
        # The triggering transition is still emitted above for downstream learning.
        if shield_intervention:
            self.gathering_experience = False
            self.safe_mode = True
            reward = -2.0

        self.last_observation = obs

        return {
            self.outputs[0]: Context(
                state=obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info
            )
        }

