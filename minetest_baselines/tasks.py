import gymnasium as gym
from gymnasium.wrappers import (
    FrameStack,
    GrayScaleObservation,
    ResizeObservation,
    TimeLimit,
)
from minetester.minetest_env import Minetest

from minetest_baselines.wrappers import (
    AlwaysDig,
    DictToMultiDiscreteActions,
    DiscreteMouseAction,
    FlattenMultiDiscreteActions,
    PenalizeJumping,
    SelectKeyActions,
    ToFloat32Reward,
)


def wrapped_treechop_env(**kwargs):
    env = Minetest(
        **kwargs,
    )
    env = TimeLimit(env, 500)
    # simplify mouse actions
    env = DiscreteMouseAction(
        env,
        num_mouse_bins=3,
        max_mouse_move=0.05,
        quantization_scheme="linear",
    )
    # make breaking blocks easier to learn
    env = AlwaysDig(env)
    # only allow basic movements
    env = SelectKeyActions(env, select_keys={"FORWARD", "JUMP"})
    # jumping usually interrupts progress towards
    # breaking nodes; apply penalty to learn faster
    env = PenalizeJumping(env, 0.01)
    # transform into pure discrete action space
    env = DictToMultiDiscreteActions(env)
    env = FlattenMultiDiscreteActions(env)
    # simplify observations
    env = ResizeObservation(env, (64, 64))
    env = GrayScaleObservation(env, keep_dim=False)
    # facilitate learning dynamics
    env = FrameStack(env, 4, lz4_compress=True)
    # cast rewards to float32
    env = ToFloat32Reward(env)
    return env


TASKS = [
    ("treechop", 0, wrapped_treechop_env),
    ("treechop", 1, wrapped_treechop_env),
    ("treechop_shaped", 0, wrapped_treechop_env),
]


for task, version, entry_point in TASKS:
    gym.register(
        f"minetester-{task}-v{version}",
        entry_point=f"{entry_point.__module__}:{entry_point.__name__}",
        kwargs=dict(clientmods=[f"{task}_v{version}"], render_mode="rgb_array"),
    )
