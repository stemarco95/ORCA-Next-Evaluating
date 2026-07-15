from typing import Dict
import numpy as np
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3 import TD3
import torch
from core.base_module import BaseModule
from utils.context import Context

MODEL_KEY = "model"
STATE_KEY = "observation"
UTILITY_KEY = "utility"
ENABLE_KEY = "model_selection"  # Key from ModelCycler that indicates if learning should be enabled
OUTPUT_KEY = "model_update"

class Learner(BaseModule):
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

        self.acceptance_threshold = self.config.get("acceptance_threshold")
        
        # Define the default noise to pass to the actor when training
        self.base_noise_std = self.config.get("noise_std", 0.1)

        # --- TD3 Configurable Hyperparameters ---
        self.learning_starts = self.config.get("learning_starts", 100)
        self.train_freq = self.config.get("train_freq", 80)           
        self.gradient_steps = self.config.get("gradient_steps", 1)   
        self.batch_size = self.config.get("batch_size", 256)         
        self.learning_rate = self.config.get("learning_rate", 3e-4)

        self.failure_step = self.config.get("failure_step", None)  # Optional step at which a failure occurs, for logging purposes

        self.reset_model = self.config.get("reset_model", None)  # Whether to reset the model (i.e., reload from initial state) when utility is restored after a disturbance
        if self.reset_model is None:
            raise ValueError("reset_model parameter must be specified in the Learner config (true or false)")

        # --- State Tracking ---
        self.global_step = 0
        self.was_training = False  # Tracks if we were training in the previous cycle

        self.weight_distance = 0.0  # To track the distance the weights have moved during training (configuration effort)
        self.message_size = 0  # To track the size of the update message sent to the actor (communication effort)
        self.output_count = 0  # To track the number of steps taken, for logging purposes

        self.count_to_failure = 0

    def step(self, inputs: Dict[str, Context]) -> Dict[str, Context]:
        # 1. Retrieve the Model
        model_ctx = inputs.get(MODEL_KEY)
        if not model_ctx:
            return {} # Safe skip if model isn't ready
        model: TD3 = model_ctx.info.get("model")

        # 2. Retrieve Utility & Experience
        utility_ctx = inputs.get(UTILITY_KEY)
        utility = utility_ctx.info.get("utility") if utility_ctx else None
        
        state_ctx = inputs.get(STATE_KEY)
        experience = state_ctx.info.get("experience") if state_ctx else None

        if ENABLE_KEY in inputs:
            enable = inputs[ENABLE_KEY].info.get("enable_learning")
        else:
            enable = True  # Default to enabled if the key is missing

        if self.failure_step is not None and self.count_to_failure >= self.failure_step:
            enable = False  # Simulate a failure by disabling learning after a certain step threshold, for testing purposes
        self.count_to_failure += 1

        # -------------------------------------------------------------
        # THE STATE MACHINE LOGIC
        # -------------------------------------------------------------
        
        # Default status for logging
        status = "idle"

        utility_threshold = self.acceptance_threshold if not self.was_training else self.acceptance_threshold * 1.01  # Add hysteresis to prevent flapping

        self.output_count += 1
        if self.output_count % self.train_freq == 0:
            self.weight_distance = 0.0  # Initialize configuration effort to zero
            self.message_size = 0  # Initialize message size to zero
        

        reset_model = False  # Flag to indicate if we should reset the model (i.e., reload from initial state)

        # Condition 1: We are below threshold -> ACTIVELY TRAINING
        if enable and utility is not None and utility < utility_threshold:
            is_training = True
            if not self.was_training and self.reset_model:
                reset_model = True
            self.was_training = True
            noise_std = self.base_noise_std 
            status = "collecting"

            # Populate Replay Buffer
            if experience is not None:
                obs, next_obs, action, reward, done, info = experience
                
                if model.replay_buffer is not None:
                    model.replay_buffer.add(obs, next_obs, action, reward, done, [info])
                    
                model.num_timesteps += 1 

            self.global_step += 1
            buffer_size = model.replay_buffer.size() if model.replay_buffer else 0
            # Check if we have passed the warm-up phase AND it's a training cycle
            if model.num_timesteps > self.learning_starts and self.global_step % self.train_freq == 0:
                if model.replay_buffer is not None and model.replay_buffer.size() >= self.batch_size:
                    # --- 1. Snapshot weights BEFORE training ---
                    # We detach and clone to keep a safe copy in memory outside the computational graph
                    weights_before = [p.clone().detach() for p in model.policy.parameters()]

                    if isinstance(self.learning_rate, list):
                        scale = max(1 - utility / self.acceptance_threshold, 0)
                        lr = self.learning_rate[0] + (self.learning_rate[1] - self.learning_rate[0]) * scale
                        model.lr_schedule = FloatSchedule(lr)
                    else:
                        model.lr_schedule = FloatSchedule(self.learning_rate)

                    model.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)
                    status = f"trained_{self.gradient_steps}_steps"
                    
                    # --- 2. Calculate L2 distance AFTER training ---
                    total_delta_sq = 0.0
                    for w_before, w_after in zip(weights_before, model.policy.parameters()):
                        total_delta_sq += torch.sum((w_after.detach() - w_before) ** 2).item()
                    
                    # The square root of the sum of squared differences (Euclidean distance)
                    self.weight_distance = total_delta_sq ** 0.5
                    self.message_size = sum(p.numel() for p in model.policy.parameters()) * 32  # Assuming 32-bit floats
                    self.output_count = 0  # Reset output count after a training step to start tracking for the next message size calculation
                else:
                    status = "waiting_for_batch_size"

        # Condition 2: Utility is good -> NOT TRAINING
        else:
            is_training = False
            noise_std = 0.0
            
            if self.was_training:
                status = "flushing_buffer_and_resetting"
                print("[Learner] Utility restored! Flushing replay buffer for future disturbances.")
                self.weight_distance = 0.0
                self.message_size = 0  # Initialize message size to zero
                if model.replay_buffer is not None:
                    model.replay_buffer.reset()
                
                # Reset counters so the next disturbance gets a proper `learning_starts` warm-up
                self.global_step = 0
                model.num_timesteps = 0 
                
                # Lock the state so we only do this once
                self.was_training = False

        # 3. Broadcast the Update Message to the Actor
        return {
            OUTPUT_KEY: Context.with_info({
                "is_training": is_training,
                "reset_model": reset_model,
                "noise_std": noise_std,
                "status": status,
                "weight_distance": self.weight_distance,
                "message_size": self.message_size
            })
        }