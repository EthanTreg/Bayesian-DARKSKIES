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
from src.utils.plots import plot_performance, plot_distributions


def predict_labels(
        output_path: str,
        loader: DataLoader,
        network: NeuralNetwork | NormFlow) -> tuple[list[int], ndarray, ndarray]:
    """
    Predicts labels or distributions using the network or flow & saves the results to a file

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
        Cluster IDs, target labels and predictions
    """
    initial_time = time()
    bins = 100
    one_sig = 0.159
    two_sig = 0.023
    ids = []
    maxima = []
    targets = []
    predictions = []
    network.train(False)

    # Generate predictions
    with torch.no_grad():
        for id_batch, target, images in loader:
            ids.extend(id_batch)
            targets.extend(target.numpy())
            prediction = network.predict(images.to(get_device()[1]))
            predictions.extend(prediction.cpu().numpy())

    targets = np.array(targets)
    predictions = np.array(predictions)

    if isinstance(network, NormFlow):
        # Prediction statistics
        medians = np.median(predictions, axis=-1)
        one_stds = np.array([
            np.quantile(predictions, one_sig, axis=-1),
            np.quantile(predictions, 1 - one_sig, axis=-1),
        ]).swapaxes(0, 1)
        two_stds = np.array([
            np.quantile(predictions, two_sig, axis=-1),
            np.quantile(predictions, 1 - two_sig, axis=-1),
        ]).swapaxes(0, 1)

        # Most likely values
        for distribution in predictions:
            hist, bins = np.histogram(distribution, bins=bins)
            maxima.append(bins[np.argmax(hist)])

        maxima = np.array(maxima)
        output = np.hstack((
            np.expand_dims(ids, axis=1),
            targets,
            np.expand_dims(maxima, axis=1),
            np.expand_dims(medians, axis=1),
            one_stds,
            two_stds,
            predictions,
        ))
        header = ('IDs,Targets,Maxima,Medians,One sigma lower,One sigma upper,'
                  'Two sigma lower,Two sigma upper,Distributions')
    else:
        output = np.hstack((np.expand_dims(ids, axis=1), targets, predictions))
        header = 'IDs,Targets,Predictions'

    print(f'Parameter prediction time: {time() - initial_time:.3e} s')
    np.savetxt(output_path, output, delimiter=',', fmt='%s', header=header)

    return ids, targets, predictions


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
    net_learning_rate = config['training']['network-learning-rate']
    flow_learning_rate = config['training']['flow-learning-rate']
    val_frac = config['training']['validation-fraction']
    name = config['training']['network-name']
    net_description = config['training']['network-description']
    flow_description = config['training']['network-description']
    data_path = config['data']['data-path']
    networks_dir = config['data']['network-configs-directory']
    states_dir = config['output']['network-states-directory']

    # Fetch dataset
    dataset = DarkDataset(data_path, transform=(-1.5, 1.5))

    # Initialise network
    network = Network(
        list(dataset[0][-1].shape),
        list(dataset[0][1].shape),
        net_learning_rate,
        name,
        networks_dir,
    ).to(device)

    # Initialise flow
    flow = NSF(
        features=1,
        context=network.shapes[-2][0],
        transforms=4,
        hidden_features=(512, 512, 256, 256)
    ).to(device)
    flow.optimiser = torch.optim.Adam(flow.parameters(), lr=flow_learning_rate)
    flow.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        flow.optimiser,
        patience=5,
        factor=0.5,
        verbose=True,
    )
    flow.name = 'flow'

    # Create network training objects
    dataset.one_hot(False)
    flow = NormFlow(
        states_dir,
        (net_save, flow_save),
        torch.unique(dataset.labels),
        flow,
        network,
        net_layers=-1,
        description=flow_description,
    )
    network = NeuralNetwork(
        net_save,
        states_dir,
        network,
        description=net_description,
        loss_function=nn.CrossEntropyLoss(),
    )
    flow.train_net = True

    # Load states from previous training
    net_indices = network.load(net_load, states_dir)
    flow_indices = flow.load(states_dir, (net_load, flow_load))

    if net_indices is None:
        indices = flow_indices
    else:
        indices = net_indices

    # Initialise datasets
    loaders = data_init(dataset, batch_size=batch_size, val_frac=val_frac, indices=indices)

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
    distributions_path = config['output']['distributions-path']

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
    loaders[0].dataset.dataset.one_hot(True)
    losses = training(
        (network.epoch, net_epochs),
        loaders,
        network,
        losses=network.losses,
    )
    plot_performance(plots_dir, 'Net_Losses', 'Loss', losses[1], train=losses[0])
    plot_performance(
        plots_dir,
        'Accuracy',
        'Accuracy (%)',
        network.accuracy,
        log_y=False,
    )
    predict_labels(predictions_path, loaders[1], network)

    # Train flow
    loaders[0].dataset.dataset.one_hot(False)
    losses = training(
        (flow.flow_epoch, flow_epochs),
        loaders,
        flow,
        losses=flow.flow_losses,
    )
    plot_performance(
        plots_dir,
        'Flow_Losses',
        'Loss',
        losses[1],
        log_y=False,
        train=losses[0],
    )
    _, targets, distributions = predict_labels(distributions_path, loaders[1], flow)
    plot_distributions(plots_dir, distributions, targets.flatten())


if __name__ == '__main__':
    main()
