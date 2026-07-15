import os
import json
from typing import Dict

from stable_baselines3 import TD3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure
import torch
from core.audit_logger import AuditLogger
from core.base_module import BaseModule
from utils.context import Context

class ModelProvider(BaseModule):
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

        library_path = self.config.get("library_path", [])
        if not library_path:
            raise ValueError("Model library path must be provided in the config under the 'library_path' key.")

        self.step_count = 0
        
        with open(library_path, 'r') as f:
            self.library = json.load(f)

        self.model_key = self.config.get("initial_model", None)
        if self.model_key is None:
            self.model_key = list(self.library.keys())[0]  # Default to the first model in the library if not specified

        self.model = self._init_model(self.library[self.model_key]["path"])

    def _init_model(self, load_path):
        if not os.path.exists(load_path):
            raise ValueError(f"Model load path does not exist: {load_path}")
            
        print(f"Loading model: {load_path}")

        custom_objects = {
            "n_envs": 1,
            "replay_buffer": None,

        }

        model = TD3.load(load_path, custom_objects=custom_objects)

        silent_logger = configure(folder=None, format_strings=[])
        model.set_logger(silent_logger)
        
        # 2. Manually recreate a fresh Replay Buffer structured for exactly 1 environment
        model.replay_buffer = ReplayBuffer(
            buffer_size=2000,
            observation_space=model.observation_space,
            action_space=model.action_space,
            device=model.device,
            n_envs=1,
            optimize_memory_usage=model.optimize_memory_usage,
        )
                
        model.num_timesteps = 0
        return model
      
    def save_checkpoint(self, timestep):
        """Call this within your training loop."""
        save_path = os.path.join(self.checkpoint_dir, f"{self.current_model_key}_step_{timestep}.zip")
        self.model.save(save_path)
        print(f"Checkpoint saved: {save_path}")

    def reset(self):

        if self.model == None:
            path = self.library[self.model_key]["path"]
            self.model = self._init_model(path)

        info = {
            "model": self.model, 
            "model_key": self.model_key
        }
       
        return {self.outputs[0]: Context.with_info(info=info)}

    def step(self, inputs: Dict[str, Context]) -> Dict[str, Context]:
        if "model_update" in inputs:
            update_ctx = inputs.get("model_update")

            reset_model = update_ctx.info.get("reset_model", False)
            if reset_model:
                AuditLogger.log_event(f"Resetting model to initial state due to reset signal.")
                self.model = self._init_model(self.library[self.model_key]["path"])

        if "model_selection" in inputs:
            selection_ctx = inputs.get("model_selection")
            if selection_ctx:
                key = selection_ctx.info.get("key")

                if key == self.model_key:
                    AuditLogger.log_event(f"Model key '{key}' is already the current model. No change made.")
                elif key in self.library:
                    self.model_key = key
                    path = self.library[key]["path"]
                    self.model = self._init_model(path)
                else:
                    AuditLogger.log_event(f"Warning: Model key '{key}' not found. Keeping current model.")

        self.step_count += 1

        new_ctx = Context.with_info(info={
            "model": self.model, 
            "model_key": self.model_key
        })
        return {self.outputs[0]: new_ctx}