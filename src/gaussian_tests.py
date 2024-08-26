"""
Runs several Gaussian training cycles with different cross-sections
"""
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import src.networks as nets
from src.main import net_init, summary
from src.utils.utils import open_config
from src.utils.data import GaussianDataset, loader_init


def init(known: list[float], unknown: list[float], config: str | dict = '../config.yaml') -> tuple[
        tuple[DataLoader, DataLoader],
        nets.BaseNetwork]:
    """
    Initialises the network and dataloaders

    Parameters
    ----------
    known : list[float]
        Known classes and labels
    unknown : list[float]
        Known classes with unknown labels to pass to the Gaussian dataset
    config : string | dictionary, default = '../config.yaml'
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

    # Fetch dataset
    dataset = GaussianDataset('../data/gaussian_data_2.pkl', known, unknown)

    net = net_init(dataset, config)
    loaders = loader_init(dataset, batch_size=batch_size, val_frac=val_frac, idxs=net.idxs)
    net.idxs = dataset.idxs
    net.transform = dataset.transform
    return loaders, net


def main(config_path: str = '../config.yaml'):
    """
    Main function for testing Gaussian toy datasets

    Parameters
    ----------
    config_path : string, default = '../config.yaml'
        Path to the configuration file
    """
    _, config = open_config('main', config_path)
    net_epochs = config['training']['network-epochs']
    runs = 5
    accuracies = []
    medians = []
    means = []
    stds = []
    known = [0.1, 0.3, 0.5, 0.7, 0.9]
    unknown = [0.75]
    unseen_unknown = [0.75]

    # Initialise network
    loaders, _ = init(known, unknown, config)
    loaders_unseen, _ = init(known, unseen_unknown, config)

    for i in range(runs):
        print(f'\nRun {i + 1}/{runs}')
        network = net_init(loaders[0].dataset.dataset, config)

        # Train network
        network.training(net_epochs, loaders)
        returns = summary(network.predict(loaders_unseen[1]))

        medians.append(returns[0])
        means.append(returns[1])
        stds.append(returns[2])
        accuracies.append(returns[3])

    medians = np.stack(medians)
    means = np.stack(means)
    stds = np.stack(stds)
    accuracies = np.stack(accuracies)
    print(pd.DataFrame(
        (
            np.mean(means, axis=0),
            np.mean(stds, axis=0),
            np.mean(medians, axis=0),
            np.std(medians, axis=0)
        ),
        index=['Means', 'Cluster Stds', 'Medians', 'Stds'],
        columns=unseen_unknown + known,
    ).round(3))


if __name__ == '__main__':
    main()
