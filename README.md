# Baselines for Minetest tasks
Baseline scripts for solving Minetest tasks.

## Algorithms
- DQN
- PPO

## Tasks
- minetester-treechop-v0
- minetester-treechop-v1
- minetester-treechop_shaped-v0

## Dependencies
- minetester
- torch
- jax
- stable-baselines3

## Installation
- Clone and compile https://github.com/EleutherAI/minetest into `MINETEST_DIR`
- Clone this repo into the parent folder of `MINETEST_DIR`, enter it and run
    - `conda create -n mtb python=3.8`
    - `conda activate mtb`
    - `pip install -e .`

## Usage
Training from scratch:
```python
python -m minetest_baselines.train --algo ALGO_NAME --task TASK_NAME --SOME_PARAM ...
```
Show help for algorithm parameters:
```
python -m minetest_baselines.train --algo ALGO_NAME --help
```
When tracking experiments and uploading models, make sure to export wandb and huggingface tokens or to login using `wandb login` and `huggingface-cli login`.

## Contributing
Please raise an issue before creating a pull request.
To install developer tools run
- `pip install -e .[dev]`
- `pre-commit install`
