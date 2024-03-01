"""
Main script for DARKSKIES Bayesian neural network
"""
import os

import torch
import matplotlib as mpl
from netloader.network import Network
from torch.utils.data import DataLoader

import src.networks as nets
from src.utils import plots
from src.utils.data import DarkDataset, loader_init
from src.utils.utils import get_device, open_config, save_name


def init(config: dict | str = '../config.yaml') -> tuple[
        tuple[DataLoader, DataLoader],
        nets.BaseNetwork]:
    """
    Initialises the network and dataloaders

    Parameters
    ----------
    config : dictionary | string, default = '../config.yaml'
        Configuration dictionary or path to the configuration dictionary

    Returns
    -------
    tuple[tuple[Dataloader, Dataloader], BaseNetwork, NormFlow]
        Train & validation dataloaders, neural network and flow training objects
    """
    if isinstance(config, str):
        _, config = open_config('main', config)

    # Constants
    device = get_device()[1]

    # Load config parameters
    batch_size = config['training']['batch-size']
    save_num = config['training']['network-save']
    load_num = config['training']['network-load']
    learning_rate = config['training']['network-learning-rate']
    val_frac = config['training']['validation-fraction']
    name = config['training']['network-name']
    description = config['training']['network-description']
    data_path = config['data']['data-path']
    networks_dir = config['data']['network-configs-directory']
    states_dir = config['output']['network-states-directory']

    # Fetch dataset
    dataset = DarkDataset(
        data_path,
        ['CDM+baryons', 'SIDM0.1+baryons', 'SIDM0.3+baryons', 'SIDM1+baryons'],
    )

    # Initialise network
    if load_num:
        net = nets.load_net(load_num, states_dir, name)
        net.description = description
        net.save_path = save_name(save_num, states_dir, name)
    else:
        net = Network(
            list(dataset[0][2].shape),
            [len(torch.unique(dataset.labels))],
            learning_rate,
            name,
            networks_dir,
        ).to(device)
        net = nets.CompactClusterEncoder(
            save_num,
            states_dir,
            torch.unique(dataset.labels),
            net,
            description=description,
        )
        # flow = nets.norm_flow(
        #     dataset.latent.size(1),
        #     4,
        #     learning_rate,
        #     [60, 60]
        # )
        # network = nets.NormFlow(save_num, states_dir, flow, description=description)

    # Initialise datasets
    dataset.normalise(transform=net.transform)
    net.classes = torch.unique(dataset.labels).to(device)
    # dataset.normalise(idxs=dataset.labels != torch.min(dataset.labels), transform=network.transform)
    # network.idxs = dataset.idxs
    loaders = loader_init(dataset, batch_size=batch_size, val_frac=val_frac, idxs=net.idxs)
    net.idxs = dataset.idxs
    net.transform = dataset.transform
    return loaders, net


def main(config_path: str = '../config.yaml'):
    """
    Main function for training and analysis of the network

    Parameters
    ----------
    config_path : string, default = '../config.yaml'
        Path to the configuration file
    """
    _, config = open_config('main', config_path)

    net_epochs = config['training']['network-epochs']
    labels = config['training']['class-labels']
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
    loaders, network = init(config)

    # Train network
    network.training(net_epochs, loaders)
    plots.plot_performance(
        plots_dir,
        'Net_Losses',
        'Loss',
        network.losses[1],
        log_y=False,
        train=network.losses[0],
    )
    _, targets, predictions, _, latent = network.predict(
        loaders[1],
        path='../data/cluster_val_vd.csv',
    )
    plots.plot_clusters(plots_dir, targets, latent, labels=labels, predictions=predictions)
    plots.plot_confusion(plots_dir, labels, targets, predictions)


if __name__ == '__main__':
    main()
