import gym
from minetest_baselines import register_tasks  # noqa
from stable_baselines3 import DQN

env = gym.make(
    "minetester-treechop_shaped-v0",
    world_seed=42, 
    headless=True, 
    start_xvfb=True
)

model = DQN("CnnPolicy", env, verbose=1, buffer_size=10000)
model.learn(total_timesteps=10000, log_interval=1)
