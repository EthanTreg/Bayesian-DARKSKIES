"""
Main script for DARKSKIES Bayesian neural network
"""
import os

import torch
import numpy as np
import pandas as pd
from numpy import ndarray
from netloader.network import Network
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

import src.networks as nets
from src.utils import plots
from src.utils.data import DarkDataset, GaussianDataset, loader_init
from src.utils.utils import get_device, open_config, save_name


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
    else:
        net = Network(
            list(dataset[0][2].shape),
            [len(torch.unique(dataset.labels))],
            learning_rate,
            name,
            networks_dir,
        ).to(device)
        # flow = nets.norm_flow(
        #     1,
        #     4,
        #     learning_rate,
        #     [256, 256, 256],
        #     context=np.prod(net.check_shapes[flow_check]),
        # )
        net = nets.CompactClusterEncoder(
            save_num,
            states_dir,
            torch.unique(dataset.labels),
            net,
            unknown=len(dataset.unknown),
            # method='median',
            description=description,
            verbose='progress',
        )
        # net = nets.NormFlowEncoder(
        #     save_num,
        #     states_dir,
        #     flow,
        #     net,
        #     flow_checkpoint=flow_check,
        #     description=description,
        #     verbose='progress',
        #     train_epochs=(10, -1),
        #     classes=torch.unique(dataset.labels),
        # )

    # Initialise datasets
    net.classes = torch.unique(dataset.labels).to(device)
    net.encoder_loss = 1
    return net


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
    known = [0.05, 1]
    unknown = [
        # 'zooms'
    ]
    batch_size = config['training']['batch-size']
    val_frac = config['training']['validation-fraction']
    data_path = config['data']['data-path']
    device = get_device()[1]

    # Fetch dataset & network
    dataset = DarkDataset(
        data_path,
        [
            'CDM+baryons',
            'SIDM0.1+baryons',
            'SIDM0.3+baryons',
            'SIDM1+baryons',
            'CDM_hi+baryons',
            'CDM_low+baryons',
            'zooms',
            'flamingo',
            # 'vdSIDM+baryons',
        ],
        unknown,
    )
    # dataset = GaussianDataset(data_path, known, unknown)
    net = net_init(dataset, config)

    # Initialise datasets
    dataset.normalise(
        idxs=torch.isin(dataset.labels, torch.unique(dataset.labels)[len(unknown):]),
        transform=net.transform,
    )
    net.classes = torch.unique(dataset.labels).to(device)
    loaders = loader_init(dataset, batch_size=batch_size, val_frac=val_frac, idxs=net.idxs)
    net.idxs = dataset.idxs
    net.transform = dataset.transform
    return loaders, net


def summary(data: dict) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Generates summary stats for the trained network

    Parameters
    ----------
    data : dictionary
        Data returned from the trained network

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, ndarray]
        Medians, means, standard deviations and accuracies for the network predictions
    """
    accuracies = []
    medians = []
    means = []
    stds = []

    for class_ in np.unique(data['targets']):
        idxs = class_ == data['targets']
        medians.append(np.median(data['latent'][idxs, 0]))
        means.append(np.mean(data['latent'][idxs, 0]))
        stds.append(np.std(data['latent'][idxs, 0]))
        accuracies.append(np.count_nonzero(data['preds'][idxs] == class_) / len(idxs))

    medians = 10 ** np.stack(medians)
    means = 10 ** np.stack(means)
    stds = np.log(10) * means * np.stack(stds)
    accuracies = np.stack(accuracies)

    return medians, means, stds, accuracies


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
    loaders, network = init(config)

    # Train network
    network.training(net_epochs, loaders)

    # Plots
    plots.plot_performance(
        plots_dir,
        'Net_Losses',
        'Loss',
        network.losses[1],
        log_y=False,
        train=network.losses[0],
    )
    data = network.predict(
        loaders[1],
        path='../data/cluster_val_hydro.pkl',
    )
    # plots.plot_distributions(
    #     plots_dir,
    #     'flow_distributions',
    #     data['distributions'],
    #     y_axis=False,
    #     titles=data['targets'],
    # )
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

    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 500)
    print(pd.DataFrame(
        summary(data)[:3],
        index=['Medians', 'Means', 'Stds'],
        columns=[label.replace('\sigma=', '').replace('$', '') for label in labels],
    ).round(3))


if __name__ == '__main__':
    main()
