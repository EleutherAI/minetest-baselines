from setuptools import setup

UTILS = [
    "wandb",
    "huggingface_hub",
    "tensorboardX",
    "tensorboard",
    "opencv-python",
    "imageio-ffmpeg",
    "jax-smi",
    "protobuf==3.20.1",
    "moviepy",
]
TORCH = ["torch", "stable-baselines3"]
JAX = ["jax", "optax", "flax"]
DEV = ["pre-commit", "black", "isort", "flake8"]

setup(
    name="Minetest Baselines",
    version="0.0.1",
    description="Train agents in complex environments based on Minetest.",
    author="EleutherAI",
    author_email="",
    packages=["minetest_baselines"],
    install_requires=["minetester>=0.0.1", *UTILS, *TORCH, *JAX],
    extras_require={"dev": DEV},
)
