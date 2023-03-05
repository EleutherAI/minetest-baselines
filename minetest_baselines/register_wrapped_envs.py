import gym
from gym.wrappers import (FrameStack, GrayScaleObservation,
                          NormalizeObservation, ResizeObservation, TimeLimit)
from minetester.minetest_env import Minetest
from minetest_baselines.wrappers import (DictToMultiDiscreteActions,
                                 DiscreteMouseAction,
                                 FlattenMultiDiscreteActions, GroupKeyActions,
                                 SelectKeyActions)


def wrapped_minetest_env(**kwargs):
    env = Minetest(
        **kwargs,
    )
    env = TimeLimit(env, 1000)
    # action space wrappers
    env = DiscreteMouseAction(
        env,
        num_mouse_bins=5,
        max_mouse_move=50,
        quantization_scheme="mu_law",
        mu=5,
    )
    env = SelectKeyActions(
        env,
        select_keys={"FORWARD", "BACKWARD", "LEFT", "RIGHT", "JUMP", "DIG"}
    )
    env = GroupKeyActions(
        env,
        groups=[{"FORWARD", "BACKWARD"}, {"LEFT", "RIGHT"}]
    )
    env = DictToMultiDiscreteActions(env)
    env = FlattenMultiDiscreteActions(env)
    # observation space wrappers
    env = ResizeObservation(env, (64, 64))
    env = GrayScaleObservation(env, keep_dim=False)
    env = FrameStack(env, 4)
    # env = NormalizeObservation(env)
    return env


TASKS = [("treechop", 0), ("treechop", 1)]


for task, version in TASKS:
    gym.register(
        f"minetester-wrapped-{task}-v{version}",
        entry_point="minetest_baselines.register_wrapped_envs:wrapped_minetest_env",
        kwargs=dict(clientmods=[f"{task}_v{version}"])
    )