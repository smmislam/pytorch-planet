import sys
import pathlib
from argparse import ArgumentParser
import ruamel.yaml as yaml
from model_based_agent import ModelBasedLearner
from utils import seed_everything


def main():
    print('======== Learning Latent Dynamics For Planning From Pixels ========')

    # Parse hyper-parameters
    parser = ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    # Comment the line above and comment out the line below if you want to debug in IDE like PyCharm
    # parser.add_argument('--configs', nargs='+', default=['defaults', 'dmc'])
    args, remaining = parser.parse_known_args()
    # Update from configs.yaml
    configs = yaml.safe_load((pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
    default_params = dict()
    for name in args.configs:
        default_params.update(configs[name])
    # Update from cli
    parser = ArgumentParser()
    for key, value in default_params.items():
        parser.add_argument('--' + key, type=type(value), default=value)
    args = parser.parse_args(remaining)
    params = vars(args)

    # Seed RNGs
    seed_everything(seed=params['rng_seed'])

    # Initialize model-based agent and learn with planet
    agent = ModelBasedLearner(params=params)
    agent.collect_seed_episodes()
    agent.learn_with_planet()


if __name__ == '__main__':
    main()
