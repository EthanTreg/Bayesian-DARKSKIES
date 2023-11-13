"""
Main script for DARKSKIES Bayesian neural network
"""
import os
from time import time

import torch
import numpy as np
import matplotlib as mpl
from numpy import ndarray
from zuko.flows import NSF
from torch import nn
from torch.utils.data import DataLoader
from netloader.network import Network

from src.utils.training import training
from src.utils.data import DarkDataset, data_init
from src.utils.utils import open_config, get_device
from src.utils.networks import NeuralNetwork, NormFlow
from src.utils.plots import plot_performance, plot_param_distribution, plot_param_comparison


def predict_distributions(config: dict | str = '../config.yaml') -> tuple[list[int], ndarray]:
    """
    Predicts the probability distribution for validation data for normalising flows

    Outdated, needs updating

    Parameters
    ----------
    config : dictionary | string, default = '../config.yaml'
        Configuration dictionary or path to the configuration dictionary

    Returns
    -------
    tuple[list[integer], ndarray]
        Cluster IDs and parameter distributions
    """
    if isinstance(config, str):
        _, config = open_config('main', config)

    if config['training']['save']:
        config['training']['load'] = config['training']['save']

    (_, loader), network, flow = init(config)[2:]

    initial_time = time()
    output_path = config['output']['parameter-predictions-path']
    ids = []
    targets = []
    labels = []

    flow.eval()
    network.eval()
    network.layer_num = -1

    # Generate predictions
    with torch.no_grad():
        for id_batch, target, images in loader:
            ids.extend(id_batch)
            targets.extend(target)
            label_batch = flow(network(images.to(get_device()[1]))).sample([1000])
            label_batch = label_batch.view(images.size(0), -1).cpu()
            labels.append(label_batch)

    labels = torch.cat(labels).cpu().numpy()

    output = np.hstack((np.expand_dims(ids, axis=1), targets, labels))
    print(f'Parameter prediction time: {time() - initial_time:.3e} s')
    np.savetxt(output_path, output, delimiter=',', fmt='%s')

    return ids, labels


def predict_labels(
        output_path: str,
        loader: DataLoader,
        network: NeuralNetwork | NormFlow) -> tuple[list[int], ndarray, ndarray]:
    """
    Predicts labels using the encoder & saves the results to a file

    Parameters
    ----------
    output_path : string
        Path to save the predictions
    loader : DataLoader
        Dataset to generate predictions for
    network : NeuralNetwork | NormFlow
        Network to generate predictions

    Returns
    -------
    tuple[list[int], ndarray, ndarray]
        Cluster IDs, target labels and label predictions
    """
    initial_time = time()
    ids = []
    targets = []
    labels = []
    network.train(False)

    # Generate predictions
    with torch.no_grad():
        for id_batch, target, images in loader:
            ids.extend(id_batch)
            targets.extend(target.numpy())
            label = network.predict(images.to(get_device()[1]))
            labels.extend(label.cpu().numpy())

    targets = np.array(targets)
    labels = np.array(labels)

    output = np.hstack((np.expand_dims(ids, axis=1), targets, labels))
    print(f'Parameter prediction time: {time() - initial_time:.3e} s')
    np.savetxt(output_path, output, delimiter=',', fmt='%s')

    return ids, targets, labels


def init(config: dict | str = '../config.yaml') -> tuple[
        tuple[DataLoader, DataLoader],
        NeuralNetwork,
        NormFlow]:
    """
    Initialises the network and dataloaders

    Parameters
    ----------
    config : dictionary | string, default = '../config.yaml'
        Configuration dictionary or path to the configuration dictionary

    Returns
    -------
    tuple[tuple[Dataloader, Dataloader], NeuralNetwork, NormFlow]
        Train & validation dataloaders, neural network and flow training objects
    """
    if isinstance(config, str):
        _, config = open_config('main', config)

    # Constants
    device = get_device()[1]

    # Load config parameters
    batch_size = config['training']['batch-size']
    net_save = config['training']['network-save']
    flow_save = config['training']['flow-save']
    net_load = config['training']['network-load']
    flow_load = config['training']['flow-load']
    learning_rate = config['training']['learning-rate']
    val_frac = config['training']['validation-fraction']
    name = config['training']['network-name']
    data_path = config['data']['data-path']
    networks_dir = config['data']['network-configs-directory']
    states_dir = config['output']['network-states-directory']

    # Fetch dataset
    dataset = DarkDataset(data_path)

    # Initialise network
    network = Network(
        list(dataset[0][-1].shape),
        list(dataset[0][1].shape),
        learning_rate,
        name,
        networks_dir,
    ).to(device)

    # network = Inceptionv4(2, 3).to(device)
    # network.name = 'inceptionV4'
    # network.optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)
    # network.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     network.optimiser,
    #     factor=0.5,
    #     verbose=True,
    # )

    # Initialise flow
    flow = NSF(
        features=1,
        context=network.shapes[-2][0],
        transforms=4,
        hidden_features=(512, 512, 256, 256)
    ).to(device)
    flow.optimiser = torch.optim.Adam(flow.parameters(), lr=1e-3)
    flow.name = 'flow'

    # Create network training objects
    flow = NormFlow(states_dir, (net_save, flow_save), flow, network)
    network = NeuralNetwork(net_save, states_dir, network, nn.CrossEntropyLoss())
    flow.network.layer_num = -1

    # Load states from previous training
    net_indices = network.load(net_load, states_dir)
    flow_indices = flow.load(states_dir, (net_load, flow_load))

    if net_indices is None:
        indices = flow_indices
    else:
        indices = net_indices

    # Initialise datasets
    loaders = data_init(data_path, batch_size=batch_size, val_frac=val_frac, indices=indices)

    return loaders, network, flow


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
    flow_epochs = config['training']['flow-epochs']
    states_dir = config['output']['network-states-directory']
    plots_dir = config['output']['plots-directory']
    predictions_path = config['output']['predictions-path']

    # Create plots directory
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Create folder to save network progress
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)

    # Initialise Matplotlib display parameters
    mpl.use('qt5agg')

    # Initialise network
    loaders, network, flow = init(config)

    # Train network
    losses = training(
        (network.epoch, net_epochs),
        loaders,
        network,
        losses=network.losses,
    )
    plot_performance(plots_dir, 'Losses', 'Loss', losses[1], train=losses[0])
    plot_performance(
        plots_dir,
        'Accuracy',
        'Accuracy (%)',
        network.accuracy,
        log_y=False,
    )

    # Train flow
    losses = training(
        (flow.flow_epoch, flow_epochs),
        loaders,
        flow,
        losses=flow.flow_losses,
    )
    plot_performance(plots_dir, 'Losses', 'Loss', losses[1], train=losses[0])
    plot_performance(
        plots_dir,
        'Accuracy',
        'Accuracy (%)',
        network.accuracy,
        log_y=False,
    )

    # predict_distributions(config)

    _, targets, labels = predict_labels(predictions_path, loaders[1], network)

    plot_param_distribution(plots_dir, (labels, targets), labels=('Predicted', 'Target'))
    plot_param_comparison(plots_dir, targets, labels)


if __name__ == '__main__':
    main()
