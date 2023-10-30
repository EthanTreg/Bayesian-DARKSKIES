"""
Main script for DARKSKIES Bayesian neural network
"""
import os
import logging as log

import torch
import matplotlib as mpl
from netloader.network import Network

from src.utils.data import data_init
from src.utils.utils import open_config, get_device


def init(config: dict | str = '../config.yaml'):
    """
    Initialises the network and dataloaders

    Parameters
    ----------
    config : dictionary | string, default = '../config.yaml'
        Configuration dictionary or path to the configuration dictionary
    """
    if isinstance(config, str):
        _, config = open_config('main', config)

    # Constants
    initial_epoch = 0
    losses = ([], [])
    device = get_device()[1]

    # Load config parameters
    load_num = config['training']['load']
    batch_size = config['training']['batch-size']
    learning_rate = config['training']['learning-rate']
    networks_dir = config['training']['networks-configs-directory']
    data_path = config['data']['data-path']
    states_dir = config['output']['network-states-directory']

    # Load previous state if provided
    if load_num:
        try:
            state = torch.load(f'{states_dir}network_{load_num}.pth', map_location=device)
            indices = state['indices']
        except FileNotFoundError:
            log.warning(f'{states_dir}network_{load_num}.pth dose not exist\n'
                        f'No state will be loaded')
            load_num = 0
            indices = None
    else:
        indices = None

    # Initialise datasets
    loaders = data_init(data_path, batch_size=batch_size, indices=indices)

    # Initialise network
    network = Network()


def main(config_path: str = '../config.yaml'):
    """
    Main function for training and analysis of the network

    Parameters
    ----------
    config_path : string, default = '../config.yaml'
        Path to the configuration file
    """
    _, config = open_config('main', config_path)

    states_dir = config['output']['network-states-directory']
    plots_dir = config['output']['plots-directory']

    # Create plots directory
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Create folder to save network progress
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)

    # Initialise Matplotlib display parameters
    mpl.use('qt5agg')

    # Initialise network
    init_epoch, losses, loaders, network = init(config)


if __name__ == '__main__':
    main()
