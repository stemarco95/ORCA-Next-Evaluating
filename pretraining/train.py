"""
Pretraining and Fine-Tuning Script for Modular Reinforcement Learning
Trains TD3 agents for an Adaptive Cruise Control (ACC) task using Stable Baselines3.
"""

from datetime import datetime
import json
import os
import random
import shutil

import gymnasium as gym
import numpy as np
import torch
import highway_env

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy


# Register the custom environment once at import time so the training/eval
# helpers can create it without extra setup.
gym.register(
    id='acc-v0',
    entry_point='highway_env.envs.acc_env:AccEnv',
    max_episode_steps=1600,
)


def setup_experiment(name: str):
    """
    Create the log and checkpoint directories for a specific experiment run.
    """
    base = "./experiments"
    log_dir = os.path.join(base, name, "logs")
    model_dir = os.path.join(base, name, "checkpoints")
   
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    return log_dir, model_dir


def train_acc_td3(
        seed: int, 
        save_dir: str, 
        disturbance: dict = {},
        target_distance: float = 10.0, 
        other_speed: float = 10.0,
        ego_length: float = 5.0,
        other_length: float = 5.0,
        generalize=False,
        env_args: dict = {},
        net_arch: list = [32, 32]
):
    """
    Train a new TD3 agent from scratch for the ACC environment.
    """
    # Build a readable experiment name from the key training settings
    name = f"dist{str(target_distance).replace('.', '-')}"
    if not generalize:
        name += f"_speed{str(other_speed).replace('.', '-')}"  
    name += f"_seed{seed}"

    if disturbance:
        disturbance_name = "_".join([f"{k}{str(v).replace('.', '-')}" for k, v in disturbance.items()])
        name += f"_{disturbance_name}"

    log_dir, model_dir = setup_experiment(os.path.join(save_dir, name))
    
    # Reproducibility
    N_PROCESSES = 6
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)

    # Environment Factory
    def _env():
        env = gym.make("acc-v0", render_mode=None)
        conf = {
            "target_distance": target_distance,
            "other_speed": other_speed,
            "ego_length": ego_length,
            "other_length": other_length,
            "generalize": generalize,
            "inital_speed": 0.0,
            "distance_noise": 0.0,
        }
        conf.update(env_args)
        env.unwrapped.configure(conf)

        if disturbance:
            env.unwrapped.configure(disturbance)
            
        return Monitor(env)
    
    # Vectorize environments for faster data collection
    train_env = SubprocVecEnv([_env for _ in range(N_PROCESSES)])
    train_env.seed(seed)
    eval_env = SubprocVecEnv([_env for _ in range(N_PROCESSES)])
    eval_env.seed(seed + 1)

    # TD3 requires action noise for continuous action space exploration
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Initialize the TD3 agent
    model = TD3(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=dict(pi=net_arch, qf=net_arch)), 
        learning_rate=1e-3, 
        action_noise=action_noise,
        gamma=0.99,                             
        batch_size=256,               
        buffer_size=2_000,          
        learning_starts=1_000,       
        train_freq=(1, "step"),       
        gradient_steps=N_PROCESSES,   
        tensorboard_log=log_dir,
        verbose=1,
        seed=seed
    )
    
    # Setup Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(2_000 // N_PROCESSES, 1), 
        deterministic=True, 
        render=False,
        n_eval_episodes=20
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(2_000 // N_PROCESSES, 1),
        save_path=model_dir,
        name_prefix=name
    )

    # Execute Training
    steps = 20_000
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    model.learn(
        total_timesteps=steps,
        callback=[eval_callback],
        tb_log_name=f"{name}_{timestamp}"
    )

    print("[INFO] Training Complete. Saving final TD3 model.")
    model.save(os.path.join(model_dir, f"{name}_final"))

    # Cleanup
    train_env.close()
    eval_env.close()


def finetune(
        seed: int, 
        save_dir: str, 
        disturbance: dict = {},
        target_distance: float = 10.0, 
        other_speed: float = 10.0,
        generalize=False,
        env_args: dict = {},
        model_path: str = "",
        name: str = ""
):
    """
    Load an existing TD3 agent and fine-tune it on a specific environment configuration.
    """
    # Derive a run name if not explicitly provided
    if not name:
        name = f"acc_dist{str(target_distance).replace('.', '-')}_speed{str(other_speed).replace('.', '-')}_seed{seed}"
        if disturbance:
            disturbance_name = "_".join([f"{k}{str(v).replace('.', '-')}" for k, v in disturbance.items()])
            name += f"_{disturbance_name}"

    log_dir, model_dir = setup_experiment(os.path.join(save_dir, name))
    
    # Save disturbance configuration for reference
    with open(os.path.join("./experiments", save_dir, name, "disturbance.json"), "w+") as f:
        json.dump(disturbance, f)

    # Reproducibility
    N_PROCESSES = 6
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)

    # Environment Factory
    def _env():
        env = gym.make("acc-v0", render_mode=None)
        conf = {
            "target_distance": target_distance,
            "other_speed": other_speed,
            "inital_speed": 0.0,
            "distance_noise": 0.0,
            "generalize": generalize,
        }
        conf.update(env_args)
        env.unwrapped.configure(conf)

        if disturbance:
            env.unwrapped.configure(disturbance)
            
        return Monitor(env)
    
    # Vectorize environments
    train_env = SubprocVecEnv([_env for _ in range(N_PROCESSES)])
    train_env.seed(seed)
    eval_env = SubprocVecEnv([_env for _ in range(N_PROCESSES)])
    eval_env.seed(seed + 1)

    # Resume from the provided checkpoint
    model = TD3.load(model_path, env=train_env)
    model.tensorboard_log = log_dir
    model.verbose = 1
    model.seed = seed
    
    # Setup Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(1_000 // N_PROCESSES, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=20
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1_000 // N_PROCESSES, 1),
        save_path=model_dir,
        name_prefix=name
    )

    # Execute Fine-tuning
    steps = 20_000
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    model.learn(
        total_timesteps=steps,
        callback=[eval_callback],
        tb_log_name=f"{name}_{timestamp}"
    )

    print("[INFO] Fine-Tuning Complete. Saving final model.")
    model.save(os.path.join(model_dir, f"{name}_final"))

    # Cleanup
    train_env.close()
    eval_env.close()


def best_model(base_dir: str, env_args: dict = {}):
    """
    Scans a directory of experiment runs, evaluates all 'best_model.zip' checkpoints,
    and isolates the overall highest-performing model.
    """
    best_path = ""
    best_reward = -float("inf")
    infos = []
    
    # Standardize evaluation environment
    env = gym.make("acc-v0")
    env.unwrapped.configure({
        "initial_speed": [0.0, 2.0],
        "distance_noise": 1.0,
    })
    env.unwrapped.configure(env_args)

    print(f"Scanning directory: {base_dir}")

    # Evaluate every discovered model
    for subdir in os.listdir(base_dir):
        full_subdir_path = os.path.join(base_dir, subdir)
        
        if os.path.isdir(full_subdir_path):
            model_path = os.path.join(full_subdir_path, "checkpoints", "best_model.zip")
            
            if os.path.exists(model_path):
                print(f"\nEvaluating model in: {subdir}")
                try:
                    model = TD3.load(model_path, env=env)
                    
                    mean_reward, std_reward = evaluate_policy(
                        model, 
                        env, 
                        n_eval_episodes=10, 
                        deterministic=True
                    )
                    
                    print(f"  Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                    
                    # Track highest mean reward
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        best_path = model_path
                        print(f"  -> New Best Model Found!")

                    infos.append({
                        "subdir": subdir,
                        "model_path": model_path,
                        "mean_reward": mean_reward,
                        "std_reward": std_reward                    
                    })
                        
                except Exception as e:
                    print(f"  [Error] Failed to evaluate {subdir}: {e}")

    # Export best model and metadata summary
    if best_path:
        print(f"\n=========================================")
        print(f"Search Complete. Best Model: {best_path}")
        print(f"Best Reward: {best_reward:.2f}")
        print(f"=========================================")
        
        dest_model_path = os.path.join(base_dir, "best_model.zip")
        metadata_path = os.path.join(base_dir, "best_model_info.json")
        
        shutil.copy2(best_path, dest_model_path)
        
        metadata = {
            "original_path": best_path,
            "best_mean_reward": float(best_reward),
            "evaluation_episodes": 10,
            "all": infos
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Model successfully copied to: {dest_model_path}")
        print(f"Metadata saved to: {metadata_path}")
        
    else:
        print("\nNo valid 'best_model.zip' files found in the specified directory.")
    
    env.close()
    return best_path, best_reward


# Main Execution Block
if __name__ == "__main__1":
    # Shared environment configuration
    conf = {
        "distance_noise": 1.0, 
        "initial_speed": [0.0, 2.0],
    }

    output_dir = "./pretraining/experiments"

    # ==========================================
    # Phase 1: Train Generalized Agent
    # ==========================================
    setting = "speed"
    for seed in [42, 43, 44]:
        train_acc_td3(seed=seed, save_dir=f"{setting}/generalize", disturbance={}, generalize=True, env_args=conf)
    
    env_args = conf.copy()
    env_args["generalize"] = True
    best_model_path, best_reward = best_model(f"{output_dir}/{setting}/generalize", env_args=env_args)

    # ==========================================
    # Phase 2: Fine-Tune on Target Distances
    # ==========================================
    setting = "target_distance"
    base_model_path = f"{output_dir}/speed/generalize/best_model.zip"
    target_distances = [6.0, 8.0, 10.0, 12.0, 14.0]
    
    for target_distance in target_distances:
        for seed in [42, 43, 44]:
            finetune(
                seed=seed, 
                save_dir=f"{setting}/{str(target_distance).replace('.', '-')}", 
                target_distance=target_distance, 
                generalize=True, 
                env_args=conf, 
                model_path=base_model_path
            )
        
        # Evaluate and isolate the best fine-tuned model for this distance
        env_args = conf.copy()
        env_args["target_distance"] = target_distance
        env_args["generalize"] = True
        best_model_path, best_reward = best_model(f"{output_dir}/{setting}/{str(target_distance).replace('.', '-')}", env_args=env_args)

    # ==========================================
    # Phase 3: Train Specialized Speed Agents
    # ==========================================
    setting = "speed"
    speeds = np.linspace(14.0, 4.0, num=21) 
    
    for speed in speeds:
        for seed in [42, 43, 44]:
            train_acc_td3(
                seed=seed, 
                save_dir=f"{setting}/speed{str(speed).replace('.', '-')}", 
                disturbance={}, 
                other_speed=speed, 
                env_args=conf
            )
        
        # Evaluate and isolate the best specialized model for this speed
        env_args = conf.copy()
        env_args["other_speed"] = speed
        env_args["generalize"] = False
        best_model_path, best_reward = best_model(f"{output_dir}/{setting}/speed{str(speed).replace('.', '-')}", env_args=env_args)