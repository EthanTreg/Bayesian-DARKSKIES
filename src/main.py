"""
Main script for DARKSKIES Bayesian neural network
"""
import os

import torch
import numpy as np
import pandas as pd
import netloader.networks as nets
from sklearn.decomposition import PCA
from netloader.network import Network
from netloader.utils import transforms
from netloader.utils.utils import get_device, save_name
from torch.utils.data import DataLoader

from src.utils import plots
from src.utils.analysis import summary
from src.utils.utils import open_config
from src.utils.data import DarkDataset, loader_init
from src.utils.clustering import CompactClusterEncoder


def net_init(dataset: DarkDataset, config: str | dict = '../config.yaml') -> nets.BaseNetwork:
    """
    Initialises the network

    Parameters
    ----------
    dataset : DarkDataset
        Dataset with inputs and outputs
    config : string | dictionary, default = '../config.yaml'
        Configuration dictionary or path to the configuration dictionary

    Returns
    -------
    BaseNetwork
        Constructed network
    """
    if isinstance(config, str):
        _, config = open_config('main', config)

    # Load config parameters
    save_num = config['training']['network-save']
    load_num = config['training']['network-load']
    learning_rate = config['training']['network-learning-rate']
    name = config['training']['network-name']
    description = config['training']['network-description']
    networks_dir = config['data']['network-configs-directory']
    states_dir = config['output']['network-states-directory']
    device = get_device()[1]

    # Initialise network
    if load_num:
        net = nets.load_net(load_num, states_dir, name)
        net.description = description
        net.save_path = save_name(save_num, states_dir, name)
        transform = net.in_transform
        param_transform = net.header['targets']
    else:
        transform = transforms.NumpyTensor()
        param_transform = transforms.MultiTransform([
            transforms.NumpyTensor(),
            transforms.Log(),
        ])
        param_transform.append(transforms.Normalise(param_transform(dataset.labels[np.isin(
            dataset.labels,
            np.unique(dataset.labels)[len(dataset.unknown):],
        )]), mean=False))
        net = Network(
            name,
            networks_dir,
            list(dataset[0][2].shape),
            [len(np.unique(dataset.labels))],
        )
        net = CompactClusterEncoder(
            save_num,
            states_dir,
            torch.unique(param_transform(dataset.labels)),
            net,
            unknown=len(dataset.unknown),
            learning_rate=learning_rate,
            description=description,
            verbose='progress',
            transform=param_transform,
            in_transform=transform,
        )

    # Initialise datasets
    dataset.images = transform(dataset.images)
    dataset.labels = param_transform(dataset.labels)
    net.classes = torch.unique(dataset.labels)
    net.encoder_loss = 1
    return net.to(device)


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

    # Load config parameters
    batch_size = config['training']['batch-size']
    val_frac = config['training']['validation-fraction']
    data_dir = config['data']['data-dir']

    # Fetch dataset & network
    unknown = ['zooms0.05']
    dataset = DarkDataset(
        data_dir,
        [
            'CDM_hi+baryons',
            'CDM+baryons',
            'CDM_low+baryons',
            'SIDM0.1+baryons',
            'SIDM0.3+baryons',
            'SIDM1+baryons',
            'zooms0.05',
            'zooms0.1',
            # 'flamingo',
            # 'vdSIDM+baryons',
        ],
        unknown,
    )
    net = net_init(dataset, config)

    # Initialise data loaders
    loaders = loader_init(dataset, batch_size=batch_size, val_frac=val_frac, idxs=net.idxs)
    net.idxs = dataset.idxs
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

    # Initialise network
    loaders, net = init(config)

    # Train network
    net.training(net_epochs, loaders)

    # Plots
    plots.plot_performance(
        plots_dir,
        'Net_Losses',
        'Loss',
        net.losses[1],
        log_y=False,
        train=net.losses[0],
    )

    # Generate predictions
    data = net.predict(loaders[1])
    data['targets'] = data['targets'].squeeze()

    # Plot predictions
    plots.plot_clusters(
        f'{plots_dir}PCA',
        data['targets'],
        PCA(n_components=4).fit_transform(data['latent']),
        labels=labels,
    )
    plots.plot_clusters(
        f'{plots_dir}Clusters',
        data['targets'],
        data['latent'],
        labels=labels,
        predictions=data['preds'],
    )
    plots.plot_confusion(plots_dir, labels, data['targets'], data['preds'])

    # Plot Saliency
    saliency = net.saliency(loaders[1], net)
    plots.plot_saliency(
        f'{plots_dir}Mass_',
        saliency['inputs'][0, 0],
        saliency['saliencies'][0, :, 0],
    )
    plots.plot_saliency(
        f'{plots_dir}Gas_',
        saliency['inputs'][0, 1],
        saliency['saliencies'][0, :, 1],
    )

    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 500)
    data['latent'][:, 0] = net.header['targets'](data['latent'][:, 0], back=True)
    print(pd.DataFrame(
        summary(data)[:3],
        index=['Medians', 'Means', 'Stds'],
        columns=[label.replace(r'\sigma=', '').replace('$', '') for label in labels],
    ).round(3))


if __name__ == '__main__':
    main()
