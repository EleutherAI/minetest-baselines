# Baselines for Minetest tasks
Baseline scripts for solving Minetest tasks.

## Algorithms
Implemented:
- DQN

Planned:
- PPO
- Recurrent PPO with LSTM or TransformerXL
- DreamerVX


## Tasks
Implemented:
- Treechop

Planned:
- collect a diverse set of items
- everything else

## Dependencies
- Minetest
- cleanRL
- PyTorch: stable-baselines3
- jax: rlax

## Installation
- Clone and compile https://github.com/EleutherAI/minetest into `MINETEST_DIR`
- Clone this repo and run
    - `conda create -f environment.yml`
    - `conda activate minetest-baselines`

## Usage
- Training from scratch
```python
python train.py --algo ALGO_NAME --task TASK_NAME --minetest MINETEST_DIR
```
- Evaluate pretrained models
```python
python eval.py --algo ALGO_NAME --task TASK_NAME
```

## Results

TODO


## Roadmap

- [ ] Test cleanRL DQN training scripts and adapt to our needs
- [ ] Add custom algorithms and more tasks