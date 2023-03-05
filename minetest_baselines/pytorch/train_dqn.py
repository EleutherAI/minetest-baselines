import gym
from minetest_baselines import register_wrapped_envs  # noqa
from stable_baselines3 import DQN

env = gym.make("minetester-wrapped-treechop-v0", seed=1, xvfb_headless=True)

model = DQN("CnnPolicy", env, verbose=1, buffer_size=10000)
model.learn(total_timesteps=10000, log_interval=1)

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
env.close()
