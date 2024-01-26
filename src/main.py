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
from netloader.network import Network
from torch.utils.data import DataLoader

import src.networks as nets
from src.utils.data import DarkDataset, loader_init
from src.utils.utils import open_config, get_device
from src.utils.plots import plot_performance, plot_distributions, plot_param_comparison


def predict_labels(
        output_path: str,
        loader: DataLoader,
        network: nets.BaseNetwork) -> tuple[list[int], ndarray, ndarray]:
    """
    Predicts labels or distributions using the network or flow & saves the results to a file

    Parameters
    ----------
    output_path : string
        Path to save the predictions
    loader : DataLoader
        Dataset to generate predictions for
    network : BaseNetwork
        Network to generate predictions

    Returns
    -------
    tuple[list[int], ndarray, ndarray]
        Cluster IDs, target labels and predictions
    """
    initial_time = time()
    bins = 100
    ids = []
    probs = []
    maxima = []
    targets = []
    predictions = []
    transform = loader.dataset.dataset.transform
    network.train(False)

    # Generate predictions
    with torch.no_grad():
        for id_batch, target, images in loader:
            ids.extend(id_batch)
            targets.extend(target.numpy())
            prediction = network.predict(images.to(get_device()[1]))
            predictions.extend(prediction.cpu().numpy())

    # Transform values
    targets = np.array(targets) * transform[1] + transform[0]
    predictions = np.array(predictions) * transform[1] + transform[0]

    if isinstance(network, nets.NormFlow):
        # Prediction statistics
        medians = np.median(predictions, axis=-1)

        # Most likely values and probabilities
        for target, distribution in zip(targets, predictions):
            hist, bins = np.histogram(distribution, bins=bins, density=True)
            prob = hist * (bins[1] - bins[0])
            probs.append(prob[np.digitize(target, bins) - 1])
            maxima.append(bins[np.argmax(hist)])

        output = np.hstack((
            np.expand_dims(ids, axis=1),
            targets,
            np.expand_dims(probs, axis=1),
            np.expand_dims(maxima, axis=1),
            np.expand_dims(medians, axis=1),
            predictions,
        ))
        header = 'IDs,Targets,Probabilities,Maxima,Medians,Distributions'
    else:
        output = np.hstack((np.expand_dims(ids, axis=1), targets, predictions))
        header = 'IDs,Targets,Predictions'

    print(f'Parameter prediction time: {time() - initial_time:.3e} s')
    np.savetxt(output_path, output, delimiter=',', fmt='%s', header=header)

    return ids, targets, predictions


def init(config: dict | str = '../config.yaml') -> tuple[
        tuple[DataLoader, DataLoader],
        nets.BaseNetwork,
        nets.NormFlow]:
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
    dataset = DarkDataset(
        data_path,
        ['CDM+baryons', 'SIDM0.1+baryons', 'SIDM0.3+baryons', 'SIDM1+baryons'],
    )

    # Initialise network
    network = Network(
        list(dataset[0][2].shape),
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
        hidden_features=(512, 512, 512, 512)
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
    flow = nets.NormFlow(
        states_dir,
        (net_save, flow_save),
        flow,
        network,
        net_layers=-1,
        description=flow_description,
    )
    flow.train_net = True
    network = nets.ClusterEncoder(
        net_save,
        states_dir,
        torch.unique(dataset.labels),
        network,
        description=net_description,
    )

    # Load states from previous training
    network.load(net_load, states_dir)
    flow.load(states_dir, (net_load, flow_load))

    if flow.idxs is not None:
        idxs = flow.idxs
    else:
        idxs = network.idxs

    # Initialise datasets
    dataset.normalise(
        idxs=torch.argwhere(dataset.labels != torch.min(dataset.labels)).flatten(),
        transform=network.transform or flow.transform,
    )
    network.init_clusters(torch.unique(dataset.labels))
    loaders = loader_init(dataset, batch_size=batch_size, val_frac=val_frac, idxs=idxs)
    network.idxs = dataset.idxs
    flow.idxs = dataset.idxs
    network.transform = dataset.transform
    flow.transform = dataset.transform
    # torch.autograd.set_detect_anomaly(True)

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
    network.training(net_epochs, loaders)
    plot_performance(
        plots_dir,
        'Net_Losses',
        'Loss',
        network.losses[1],
        train=network.losses[0],
    )
    _, targets, predictions = predict_labels(predictions_path, loaders[1], network)
    plot_param_comparison(plots_dir, targets, predictions)
    plot_distributions(
        plots_dir,
        'Network_Distribution',
        targets[np.newaxis],
        labels=['Target', 'Prediction'],
        data_twin=predictions[np.newaxis],
    )

    # Train flow
    flow.training(flow_epochs, loaders)
    plot_performance(
        plots_dir,
        'Flow_Losses',
        'Loss',
        flow.losses[1],
        log_y=False,
        train=flow.losses[0],
    )
    _, targets, distributions = predict_labels(distributions_path, loaders[1], flow)

    data_range = [flow.transform[0], flow.transform[0] + flow.transform[1]]
    data_range[0] -= 0.1 * (data_range[1] - data_range[0])
    data_range[1] += 0.1 * (data_range[1] - data_range[0])
    plot_distributions(
        plots_dir,
        'Flow_Distributions',
        distributions,
        y_axis=False,
        labels=['Predictions', 'Target'],
        hist_kwargs={'range': data_range},
        data_twin=targets,
    )


if __name__ == '__main__':
    main()
