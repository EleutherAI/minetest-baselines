from setuptools import setup

UTILS = ['wandb', 'tensorboardX', 'opencv-python']
TORCH = ['torch', 'stable-baselines3']
JAX = ['jax', 'optax', 'flax']

setup(
    name='Minetest Baselines',
    version='0.0.1',
    description='Train agents in complex environments based on Minetest.',
    author='EleutherAI',
    author_email='',
    packages=['minetest_baselines'],
    install_requires=[
        'minetester>=0.0.1',
        *UTILS, *TORCH, *JAX
    ],
)
