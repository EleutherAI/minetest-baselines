# adapted from cleanRL: https://github.com/vwxyzjn/cleanrl
import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Callable
import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import psutil
from flax.training.train_state import TrainState
from jax_smi import initialise_tracking
from stable_baselines3.common.buffers import ReplayBuffer
from tensorboardX import SummaryWriter

import minetest_baselines.tasks  # noqa

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="minetest-baselines",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="the entity (team) of wandb's project",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to capture videos of the agent behavior",
    )
    parser.add_argument(
        "--video-frequency",
        type=int,
        default=100,
        help="number of episodes between video recordings",
    )
    parser.add_argument(
        "--save-model",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to save model into the `runs/{run_name}` folder",
    )
    parser.add_argument(
        "--upload-model",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to upload the saved model to huggingface",
    )
    parser.add_argument(
        "--hf-entity",
        type=str,
        default="",
        help="the user or org name of the model repository from the Hugging Face Hub",
    )

    # Algorithm specific arguments
    parser.add_argument(
        "--env-id",
        type=str,
        default="minetester-treechop_shaped-v0",
        help="the id of the environment",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1000000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=500000,
        help="the replay memory buffer size",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="the discount factor gamma",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="the target network update rate",
    )
    parser.add_argument(
        "--target-network-frequency",
        type=int,
        default=10000,
        help="the timesteps it takes to update the target network",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="the batch size of sample from the reply memory",
    )
    parser.add_argument(
        "--start-e",
        type=float,
        default=1,
        help="the starting epsilon for exploration",
    )
    parser.add_argument(
        "--end-e",
        type=float,
        default=0.01,
        help="the ending epsilon for exploration",
    )
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.9,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=5000,
        help="timestep to start learning",
    )
    parser.add_argument(
        "--train-frequency",
        type=int,
        default=10,
        help="the frequency of training",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="the number of environments to sample from",
    )
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    return args


def make_env(env_id, seed, idx, capture_video, video_frequency, run_name):
    def thunk():
        env = gym.make(
            env_id,
            base_seed=seed + idx,
            headless=True,
            start_xvfb=False,
            env_port=5555 + idx,
            server_port=30000 + idx,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env,
                    f"videos/{run_name}",
                    lambda x: x % video_frequency == 0,
                    name_prefix=f"env-{idx}",
                    disable_logger=True,
                )
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    epsilon: float = 0.05,
    capture_video: bool = True,
    video_frequency: int = 2,
    seed: int = 1,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, capture_video, video_frequency, run_name)],
    )
    obs, _ = envs.reset()
    model = Model(action_dim=envs.single_action_space.n)
    q_key = jax.random.PRNGKey(seed)
    params = model.init(q_key, obs)
    with open(model_path, "rb") as f:
        params = flax.serialization.from_bytes(params, f.read())
    model.apply = jax.jit(model.apply)

    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)],
            )
        else:
            q_values = model.apply(params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos.keys():
            episodic_return = infos["final_info"][0]["episode"]["r"]
            episodic_length = infos["final_info"][0]["episode"]["l"]
            print(
                f"eval_episode={len(episodic_returns)}, "
                f"episodic_return={episodic_return[0]}, "
                f"episodic_length={episodic_length[0]}",
            )
            episodic_returns += [episodic_return]
        obs = next_obs

    return episodic_returns


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # shape = (batch, stack, img_x, img_y)
        batch_dim = x.shape[0]
        x = x / 255.0
        x = jnp.transpose(x, (0, 2, 3, 1))  # shape = (batch, img x, img y, stack)
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((batch_dim, -1))  # shape = (batch, output features)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def train(args=None):
    if args is None:
        args = parse_args()
    else:
        args = parse_args(args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed,
                i,
                args.capture_video,
                args.video_frequency,
                run_name,
            )
            for i in range(args.num_envs)
        ],
    )
    assert isinstance(
        envs.single_action_space,
        gym.spaces.Discrete,
    ), "only discrete action space is supported"

    obs, _ = envs.reset()

    q_network = QNetwork(action_dim=envs.single_action_space.n)

    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, obs),
        target_params=q_network.init(q_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    q_network.apply = jax.jit(q_network.apply)
    # This step is not necessary as init called on same observation
    # and key will always lead to same initializations
    q_state = q_state.replace(
        target_params=optax.incremental_update(
            q_state.params,
            q_state.target_params,
            1,
        ),
    )

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        n_envs=args.num_envs,
        handle_timeout_termination=True,
    )

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        q_next_target = q_network.apply(
            q_state.target_params,
            next_observations,
        )  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * args.gamma * q_next_target

        def mse_loss(params):
            q_pred = q_network.apply(params, observations)  # (batch_size, num_actions)
            q_pred = q_pred[
                np.arange(q_pred.shape[0]),
                actions.squeeze(),
            ]  # (batch_size,)
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            q_state.params,
        )
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state

    initialise_tracking()
    start_time = time.time()

    def info_dict_to_array(info: dict):
        dim = list(info.values())[0].shape[0]
        ar = [{} for _ in range(dim)]
        for key, value_ar in info.items():
            for idx, element in enumerate(value_ar):
                ar[idx][key] = element
        return ar

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)],
            )
        else:
            q_values = q_network.apply(q_state.params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, truncated, infos = envs.step(actions)
        infos.update({"TimeLimit.truncated": truncated, "terminated": dones})
        array_infos = info_dict_to_array(infos)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos.keys():
            # compute mean episodic return and length across envs
            final_infos = infos["final_info"]
            mean_episodic_return = np.mean(
                [final_info["episode"]["r"] for final_info in final_infos],
            )
            mean_episodic_length = np.mean(
                [final_info["episode"]["l"] for final_info in final_infos],
            )
            print(
                f"global_step={global_step},"
                f"mean_episodic_return={mean_episodic_return}",
            )
            writer.add_scalar(
                "charts/episodic_return",
                mean_episodic_return,
                global_step,
            )
            writer.add_scalar(
                "charts/episodic_length",
                mean_episodic_length,
                global_step,
            )
            writer.add_scalar("charts/epsilon", epsilon, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(
            obs,
            real_next_obs,
            actions,
            rewards,
            dones,
            array_infos,
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                # perform a gradient-descent step
                loss, old_val, q_state = update(
                    q_state,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                    data.rewards.flatten().numpy(),
                    data.dones.flatten().numpy(),
                )

                if global_step % 100 == 0:
                    writer.add_scalar(
                        "losses/td_loss",
                        jax.device_get(loss),
                        global_step,
                    )
                    writer.add_scalar(
                        "losses/q_values",
                        jax.device_get(old_val).mean(),
                        global_step,
                    )
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

            # update target network
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(
                        q_state.params,
                        q_state.target_params,
                        args.tau,
                    ),
                )

    # Close training envs
    envs.close()
    # kill any remaining minetest processes
    for proc in psutil.process_iter():
        if proc.name() in ["minetest"]:
            proc.kill()

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(q_state.params))
        print(f"model saved to {model_path}")

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            epsilon=0.05,
            seed=args.seed,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from minetest_baselines.utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "DQN",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    writer.close()


if __name__ == "__main__":
    train()
