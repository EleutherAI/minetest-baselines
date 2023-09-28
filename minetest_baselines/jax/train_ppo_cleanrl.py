# adapted from cleanRL: https://github.com/vwxyzjn/cleanrl
import argparse
import gc
import os
import random
import time
from distutils.util import strtobool
from functools import partial
from typing import Callable, Sequence

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import psutil
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax_smi import initialise_tracking
from tensorboardX import SummaryWriter

import minetest_baselines.tasks  # noqa

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
# see https://github.com/google/jax/discussions/6332#discussioncomment-1279991


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
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )
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
        default=5000000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="the discount factor gamma",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=4,
        help="the number of mini-batches",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=4,
        help="the K epochs to update the policy",
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.1,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="coefficient of the entropy",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="coefficient of the value function",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="the target KL divergence threshold",
    )
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
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
    capture_video: bool = True,
    video_frequency: int = 2,
    seed: int = 1,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, capture_video, video_frequency, run_name)],
    )
    Network, Actor, Critic = Model
    # next_obs, _ = envs.reset()
    network = Network()
    actor = Actor(action_dim=envs.single_action_space.n)
    critic = Critic()
    key = jax.random.PRNGKey(seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    network_params = network.init(
        network_key,
        np.array([envs.single_observation_space.sample()]),
    )
    actor_params = actor.init(
        actor_key,
        network.apply(
            network_params,
            np.array([envs.single_observation_space.sample()]),
        ),
    )
    critic_params = critic.init(
        critic_key,
        network.apply(
            network_params,
            np.array([envs.single_observation_space.sample()]),
        ),
    )
    # note: critic_params is not used in this script
    with open(model_path, "rb") as f:
        (
            args,
            (network_params, actor_params, critic_params),
        ) = flax.serialization.from_bytes(
            (None, (network_params, actor_params, critic_params)),
            f.read(),
        )

    @jax.jit
    def get_action_and_value(
        network_params: flax.core.FrozenDict,
        actor_params: flax.core.FrozenDict,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        hidden = network.apply(network_params, next_obs)
        logits = actor.apply(actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/
        #     sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return action, key

    # a simple non-vectorized version
    episodic_returns = []
    for episode in range(eval_episodes):
        episodic_return = 0
        episodic_length = 0
        next_obs, _ = envs.reset()
        terminated = np.array([False])
        truncated = np.array([False])

        while not (terminated[0] or truncated[0]):
            actions, key = get_action_and_value(
                network_params,
                actor_params,
                next_obs,
                key,
            )
            next_obs, reward, terminated, truncated, infos = envs.step(
                np.array(actions),
            )
            episodic_return += reward[0]
            episodic_length += 1

            if terminated[0] or truncated[0]:
                print(
                    f"eval_episode={len(episodic_returns)}, "
                    f"episodic_return={episodic_return}, "
                    f"episodic_length={episodic_length}",
                )
                episodic_returns.append(episodic_return)

    return episodic_returns


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            x,
        )
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)


@flax.struct.dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


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
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

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

    def step_env_wrapped(action, step):
        (next_obs, reward, next_done, next_truncated, _) = envs.step(action)
        reward = reward.astype(jnp.float32)
        return next_obs, reward, next_done, next_truncated

    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = (
            1.0
            - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        )
        return args.learning_rate * frac

    network = Network()
    actor = Actor(action_dim=envs.single_action_space.n)
    critic = Critic()
    network_params = network.init(
        network_key,
        np.array([envs.single_observation_space.sample()]),
    )
    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            network_params,
            actor.init(
                actor_key,
                network.apply(
                    network_params,
                    np.array([envs.single_observation_space.sample()]),
                ),
            ),
            critic.init(
                critic_key,
                network.apply(
                    network_params,
                    np.array([envs.single_observation_space.sample()]),
                ),
            ),
        ),
        tx=optax.adam(learning_rate=args.learning_rate),
        # FIXME throws TypeError
        # tx=optax.chain(
        #    optax.clip_by_global_norm(args.max_grad_norm),
        #    optax.inject_hyperparams(optax.adam)(
        #        learning_rate=linear_schedule
        #        if args.anneal_lr else args.learning_rate,
        #        eps=1e-5
        #    ),
        # ),
    )
    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    # ALGO Logic: Storage setup
    storage = Storage(
        obs=jnp.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        ),
        actions=jnp.zeros(
            (args.num_steps, args.num_envs) + envs.single_action_space.shape,
            dtype=jnp.int32,
        ),
        logprobs=jnp.zeros((args.num_steps, args.num_envs)),
        dones=jnp.zeros((args.num_steps, args.num_envs)),
        values=jnp.zeros((args.num_steps, args.num_envs)),
        advantages=jnp.zeros((args.num_steps, args.num_envs)),
        returns=jnp.zeros((args.num_steps, args.num_envs)),
        rewards=jnp.zeros((args.num_steps, args.num_envs)),
    )

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
        step: int,
        key: jax.random.PRNGKey,
    ):
        hidden = network.apply(agent_state.params.network_params, next_obs)
        logits = actor.apply(agent_state.params.actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/
        #     sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        value = critic.apply(agent_state.params.critic_params, hidden)
        storage = storage.replace(
            obs=storage.obs.at[step].set(next_obs),
            dones=storage.dones.at[step].set(next_done),
            actions=storage.actions.at[step].set(action),
            logprobs=storage.logprobs.at[step].set(logprob),
            values=storage.values.at[step].set(value.squeeze()),
        )
        return storage, action, key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        hidden = network.apply(params.network_params, x)
        logits = actor.apply(params.actor_params, hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        value = critic.apply(params.critic_params, hidden).squeeze()
        return logprob, entropy, value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(
        compute_gae_once,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        next_value = critic.apply(
            agent_state.params.critic_params,
            network.apply(agent_state.params.network_params, next_obs),
        ).squeeze()

        advantages = jnp.zeros((args.num_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once,
            advantages,
            (dones[1:], values[1:], values[:-1], storage.rewards),
            reverse=True,
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage


    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns):
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        def update_epoch(carry, unused_inp):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_map(flatten, storage)
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)

            def update_minibatch(agent_state, minibatch):
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
                update_minibatch, agent_state, shuffled_storage
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    def rollout(
        agent_state,
        episode_stats,
        next_obs,
        next_done,
        storage,
        key,
        global_step,
    ):
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            storage, action, key = get_action_and_value(
                agent_state,
                next_obs,
                next_done,
                storage,
                step,
                key,
            )

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, next_truncated = step_env_wrapped(action, step)
            new_episode_return = episode_stats.episode_returns + reward
            new_episode_length = episode_stats.episode_lengths + 1

            episode_stats = episode_stats.replace(
                episode_returns=(
                    new_episode_return * (1 - next_done) * (1 - next_truncated)
                ).astype(jnp.float32),
                episode_lengths=(
                    new_episode_length * (1 - next_done) * (1 - next_truncated)
                ).astype(jnp.int32),
                # only update the `returned_episode_returns` if the episode is done
                returned_episode_returns=jnp.where(
                    next_done + next_truncated,
                    new_episode_return,
                    episode_stats.returned_episode_returns,
                ),
                returned_episode_lengths=jnp.where(
                    next_done + next_truncated,
                    new_episode_length,
                    episode_stats.returned_episode_lengths,
                ),
            )
            storage = storage.replace(rewards=storage.rewards.at[step].set(reward))
        return (
            agent_state,
            episode_stats,
            next_obs,
            next_done,
            storage,
            key,
            global_step,
        )


    # TRY NOT TO MODIFY: start the game
    initialise_tracking()
    global_step = 0
    envs.reset()
    # intial reset of the environments
    next_obs, _ = envs.reset()
    next_done = jnp.zeros(args.num_envs, dtype=jnp.bool_)

    start_time = time.time()
    for update in range(1, args.num_updates + 1):
        # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        update_time_start = time.time()
        (
            agent_state,
            episode_stats,
            next_obs,
            next_done,
            storage,
            key,
            global_step,
        ) = rollout(
            agent_state,
            episode_stats,
            next_obs,
            next_done,
            storage,
            key,
            global_step,
        )
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            agent_state,
            storage,
            key,
        )
        avg_episodic_return = np.mean(
            jax.device_get(episode_stats.returned_episode_returns),
        )
        avg_episodic_length = np.mean(
            jax.device_get(episode_stats.returned_episode_lengths),
        )
        print(
            f"global_step={global_step}, "
            f"avg_episodic_return={avg_episodic_return}, "
            f"avg_episodic_length={avg_episodic_length}",
        )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/avg_episodic_return",
            avg_episodic_return,
            global_step,
        )
        writer.add_scalar(
            "charts/avg_episodic_length",
            avg_episodic_length,
            global_step,
        )
        # writer.add_scalar("charts/learning_rate",
        # agent_state.opt_state[1].hyperparams["learning_rate"].item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.mean().item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.mean().item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.mean().item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.mean().item(), global_step)
        writer.add_scalar("losses/loss", loss.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS",
            int(global_step / (time.time() - start_time)),
            global_step,
        )
        writer.add_scalar(
            "charts/SPS_update",
            int(args.num_envs * args.num_steps / (time.time() - update_time_start)),
            global_step,
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
            f.write(
                flax.serialization.to_bytes(
                    [
                        vars(args),
                        [
                            agent_state.params.network_params,
                            agent_state.params.actor_params,
                            agent_state.params.critic_params,
                        ],
                    ],
                ),
            )
        print(f"model saved to {model_path}")

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Network, Actor, Critic),
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
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    writer.close()


if __name__ == "__main__":
    train()
