from typing import List, Set

import gymnasium as gym
import numpy as np
from minetester.minetest_env import Minetest
from minetester.utils import KEY_MAP, NOOP_ACTION


class MinetestWrapper(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(
            env.unwrapped,
            Minetest,
        ), "This wrapper only works on Minetest environments."
        super().__init__(env)


class AlwaysDig(MinetestWrapper):
    def step(self, action):
        action["DIG"] = True
        obs, rew, done, truncated, info = self.env.step(action)
        return obs, rew, done, truncated, info


class PenalizeJumping(MinetestWrapper):
    def __init__(self, env, jump_penalty=0.01):
        super().__init__(env)
        self.jump_penalty = jump_penalty

    def step(self, action):
        penalty = 0
        if action["JUMP"]:
            penalty = -self.jump_penalty
        obs, rew, done, truncated, info = self.env.step(action)
        rew += penalty
        return obs, rew, done, truncated, info


class FlattenMultiDiscreteActions(gym.Wrapper):
    def __init__(self, env: gym.Env):
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        super().__init__(env)
        self.action_shape = self.env.action_space.nvec
        self.action_space = gym.spaces.Discrete(np.prod(self.action_shape))

    def step(self, action):
        multidisc_action = np.unravel_index(action, self.action_shape)
        return self.env.step(np.array(multidisc_action))


class DictToMultiDiscreteActions(gym.Wrapper):
    def __init__(self, env: gym.Env):
        assert isinstance(env.action_space, gym.spaces.Dict)
        space_dims = {}
        combined_shape = []
        for key, space in env.action_space.items():
            if isinstance(space, gym.spaces.Discrete):
                combined_shape += [space.n]
                space_dims[key] = 1
            elif isinstance(space, gym.spaces.MultiDiscrete):
                combined_shape += space.nvec.tolist()
                space_dims[key] = len(space.nvec.shape)
        super().__init__(env)
        self.space_dims = space_dims
        self.action_space = gym.spaces.MultiDiscrete(combined_shape)

    def step(self, action):
        dict_action = {}
        pointer = 0
        for key, dim in self.space_dims.items():
            dict_action[key] = action[pointer : pointer + dim].squeeze()
            pointer += dim
        return self.env.step(dict_action)


class GroupKeyActions(MinetestWrapper):
    def __init__(
        self,
        env: gym.Env,
        groups: List[Set[str]],
    ):
        super().__init__(env)
        total_keys = 0
        for group in groups:
            for key in group:
                assert key in KEY_MAP, f"Selected key '{key}' not supported."
                total_keys += 1
        assert (
            len(set().union(*groups)) == total_keys
        ), "Keys can only belong to one key group!"
        self.groups = groups
        self.grouped_keys = set().union(*groups)
        key_group_spaces = {
            "^".join(group): gym.spaces.Discrete(len(group) + 1)
            for group in self.groups
        }
        self.action_space = gym.spaces.Dict(
            {
                **{
                    key: space
                    for key, space in self.env.action_space.items()
                    if key not in self.grouped_keys
                },
                **key_group_spaces,
            },
        )

    def step(self, action):
        ungrouped_action = dict(action)
        for key in action:
            if "^" in key:
                group = key.split("^")
                group_action = action[key]
                del ungrouped_action[key]
                for gidx, gkey in enumerate(group):
                    ungrouped_action[gkey] = group_action == gidx + 1
        return self.env.step(ungrouped_action)


class SelectKeyActions(MinetestWrapper):
    def __init__(
        self,
        env: gym.Env,
        select_keys: Set[str] = KEY_MAP.keys(),
    ):
        super().__init__(env)
        for key in select_keys:
            assert key in KEY_MAP, f"Selected key '{key}' is not supported."
        self.selected_keys = select_keys
        self.action_space = gym.spaces.Dict(
            {
                **{key: gym.spaces.Discrete(2) for key in self.selected_keys},
                **{"MOUSE": self.env.action_space["MOUSE"]},
            },
        )

    def step(self, action):
        full_action = NOOP_ACTION
        full_action.update(action)
        return self.env.step(full_action)


class DiscreteMouseAction(MinetestWrapper):
    def __init__(
        self,
        env: gym.Env,
        num_mouse_bins: int = 5,
        max_mouse_move: int = 50,
        quantization_scheme: str = "linear",
        mu: float = 5.0,
    ):
        super().__init__(env)
        self.max_mouse_move = max_mouse_move
        self.num_mouse_actions = num_mouse_bins**2
        self.num_mouse_bins = num_mouse_bins
        self.bin_size = 2 * self.max_mouse_move / (self.num_mouse_bins - 1)
        self.quantization_scheme = quantization_scheme
        self.mu = mu

        self.mouse_action_space = gym.spaces.Discrete(self.num_mouse_actions)
        self.action_space = gym.spaces.Dict(
            {
                **{
                    key: space
                    for key, space in self.env.action_space.items()
                    if key != "MOUSE"
                },
                **{"MOUSE": self.mouse_action_space},
            },
        )

    def discretize(self, xy):
        xy = np.clip(xy, -self.max_mouse_move, self.max_mouse_move)

        if self.quantization_scheme == "mu_law":
            xy = xy / self.max_mouse_move
            v_encode = np.sign(xy) * (
                np.log(1.0 + self.mu * np.abs(xy)) / np.log(1.0 + self.mu)
            )
            v_encode *= self.max_mouse_move
            xy = v_encode

        return np.round((xy + self.max_mouse_move) / self.bin_size).astype(np.int64)

    def undiscretize(self, xy):
        xy = xy * self.bin_size - self.max_mouse_move

        if self.quantization_scheme == "mu_law":
            xy = xy / self.max_mouse_move
            v_decode = (
                np.sign(xy) * (1.0 / self.mu) * ((1.0 + self.mu) ** np.abs(xy) - 1.0)
            )
            v_decode *= self.max_mouse_move
            xy = v_decode
        return xy

    def step(self, action):
        mouse_action = action["MOUSE"]
        if not isinstance(mouse_action, np.ndarray):
            mouse_action = np.ndarray(mouse_action)
        if mouse_action.shape == ():
            mouse_action = mouse_action[None]
        xy_action = np.concatenate(
            np.unravel_index(mouse_action, (self.num_mouse_bins, self.num_mouse_bins)),
        )
        undisc_mouse_action = self.undiscretize(xy_action)
        action["MOUSE"] = undisc_mouse_action.astype(int)
        return self.env.step(action)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = Minetest(seed=1)

    env = DiscreteMouseAction(
        env,
        num_mouse_bins=5,
        max_mouse_move=50,
        quantization_scheme="linear",
    )
    # Visualize the discretization of mouse actions
    x = np.linspace(-80, 80, 1000)  # in pixels
    x0 = np.zeros_like(x)
    x = np.vstack((x, x0))
    x_disc = env.discretize(x)
    plt.plot(x[0, :], x_disc[0, :], label="discretized")
    x_recon = env.undiscretize(x_disc)
    plt.scatter(x_recon[0, :], x_disc[0, :], label="undiscretized")
    plt.grid()
    plt.xlabel("pixels")
    plt.ylabel("bins")
    plt.legend()
    plt.show()

    # Only use a subset of the available keys
    env = SelectKeyActions(
        env,
        select_keys={"FORWARD", "BACKWARD", "LEFT", "RIGHT", "JUMP", "DIG"},
    )
    # Combine mutually exclusive actions into key groups
    env = GroupKeyActions(env, groups=[{"FORWARD", "BACKWARD"}, {"LEFT", "RIGHT"}])

    env = DictToMultiDiscreteActions(env)
    env = FlattenMultiDiscreteActions(env)

    # Render a few steps with the new action space
    env.reset()
    for _ in range(100):
        act = env.action_space.sample()
        env.step(act)
        env.render()

    env.close()
