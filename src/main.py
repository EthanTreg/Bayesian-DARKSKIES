"""
Main script for DARKSKIES Bayesian neural network
"""
import os
import logging as log
from time import time

import torch
import numpy as np
import matplotlib as mpl
from torch import Tensor
from torch.utils.data import DataLoader
from netloader.network import Network

from src.utils.data import data_init
from src.utils.training import training
from src.utils.utils import open_config, get_device, load_network, name_sort
from src.utils.plots import plot_loss, plot_param_distribution, plot_param_comparison


def init(config: dict | str = '../config.yaml') -> tuple[
    int,
    tuple[list[Tensor], list[Tensor]],
    tuple[DataLoader, DataLoader],
    Network,
]:
    """
    Initialises the network and dataloaders

    Parameters
    ----------
    config : dictionary | string, default = '../config.yaml'
        Configuration dictionary or path to the configuration dictionary

    Returns
    -------
    tuple[integer, tuple[list[Tensor], list[Tensor]], tuple[Dataloader, Dataloader], Network]
        Initial epoch; train & validation losses; train & validation dataloaders; & network
    """
    if isinstance(config, str):
        _, config = open_config('main', config)

    # Constants
    init_epoch = 0
    losses = ([], [])
    device = get_device()[1]

    # Load config parameters
    load_num = config['training']['load']
    batch_size = config['training']['batch-size']
    learning_rate = config['training']['learning-rate']
    name = config['training']['network-name']
    networks_dir = config['training']['network-configs-directory']
    data_path = config['data']['data-path']
    states_dir = config['output']['network-states-directory']

    # Load previous state if provided
    if load_num:
        try:
            state = torch.load(f'{states_dir}{name}_{load_num}.pth', map_location=device)
            indices = state['indices']
        except FileNotFoundError:
            log.warning(f'{states_dir}{name}_{load_num}.pth dose not exist\n'
                        f'No state will be loaded')
            load_num = 0
            indices = None
    else:
        indices = None

    # Initialise datasets
    loaders = data_init(data_path, batch_size=batch_size, indices=indices)

    # Initialise network
    network = Network(
        list(loaders[0].dataset[0][-1].shape),
        [1],
        learning_rate,
        name,
        networks_dir,
    ).to(device)

    # Load states from previous training
    if load_num:
        init_epoch, network, losses = load_network(load_num, states_dir, network)

    return init_epoch, losses, loaders, network


def predict_labels(config: dict | str = '../config.yaml') -> tuple[list[int], np.ndarray]:
    """
    Predicts labels using the encoder & saves the results to a file

    Parameters
    ----------
    config : dictionary | string, default = '../config.yaml'
        Configuration dictionary or file path to the configuration file

    Returns
    -------
    ndarray
        Cluster IDs and label predictions
    """
    if isinstance(config, str):
        _, config = open_config('main', config)

    if config['training']['save']:
        config['training']['load'] = config['training']['save']

    (_, loader), network = init(config)[2:]

    initial_time = time()
    output_path = config['output']['parameter-predictions-path']
    ids = []
    labels = []

    network.eval()

    # Generate predictions
    with torch.no_grad():
        for id_batch, _, images in loader:
            ids.extend(id_batch)
            label_batch = network(images.to(get_device()[1]))
            labels.append(label_batch)

    labels = torch.cat(labels).cpu().numpy()

    output = np.hstack((np.expand_dims(ids, axis=1), labels))
    print(f'Parameter prediction time: {time() - initial_time:.3e} s')
    np.savetxt(output_path, output, delimiter=',', fmt='%s')

    return ids, labels


def main(config_path: str = '../config.yaml'):
    """
    Main function for training and analysis of the network

    Parameters
    ----------
    config_path : string, default = '../config.yaml'
        Path to the configuration file
    """
    _, config = open_config('main', config_path)

    save_num = config['training']['save']
    num_epochs = config['training']['epochs']
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

    # Train network
    losses, *_ = training(
        (init_epoch, num_epochs),
        loaders, network,
        save_num=save_num,
        states_dir=states_dir,
        losses=losses,
    )
    plot_loss(plots_dir, *losses)

    ids, labels = predict_labels(config)
    val_idxs = loaders[1].dataset.indices
    target_labels = loaders[1].dataset.dataset.labels[val_idxs, 0].numpy()
    target_ids = loaders[1].dataset.dataset.ids[val_idxs]

    _, (labels, target_labels) = name_sort(
        [np.array(ids), target_ids],
        [labels, target_labels],
    )

    plot_param_distribution(plots_dir, (labels, target_labels), labels=('Predicted', 'Target'))
    plot_param_comparison(plots_dir, target_labels, labels)


if __name__ == '__main__':
    main()
