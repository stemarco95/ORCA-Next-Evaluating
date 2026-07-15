import gymnasium as gym
import highway_env # needed to register the custom environment

gym.register(
    id='acc-v0',
    entry_point='highway_env.envs.acc_env:AccEnv',
    max_episode_steps=1200,
)