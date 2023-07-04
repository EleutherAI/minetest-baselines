import argparse

from minetest_baselines.algos import ALGOS


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script", add_help=False)

    parser.add_argument("--algo", type=str, default="ppo", help="Name of the algorithm")

    parser.add_argument(
        "--task",
        type=str,
        default="minetester-treechop_shaped-v0",
        help="Name of the task",
    )

    args, algo_args = parser.parse_known_args()

    return args, algo_args


def main():
    args, algo_args = parse_arguments()
    algo = ALGOS[args.algo.lower()]
    if "--env-id" not in algo_args:
        algo_args.insert(0, "--env-id")
        algo_args.insert(1, args.task)
    algo(algo_args)


if __name__ == "__main__":
    main()
