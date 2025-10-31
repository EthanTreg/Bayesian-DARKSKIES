"""
Test script for varying latent dimensions and simulations sets in a neural network.
"""
import os
from typing import Any

import numpy as np
from numpy import ndarray
from netloader.networks import BaseNetwork

from src.batch_train import update_sims
from src.utils.utils import open_config
from src.tests.netloader_tests import TestConfig, run_tests


def change_latent_dim(
        config: dict[str, dict[str, Any] | list[dict[str, Any]]],
        *,
        latent_dim: int) -> None:
    """
    Change the latent dimension in the network JSON file given by the layer before the last
    Checkpoint layer.

    Parameters
    ----------
    config : dict[str, dict[str, Any] | list[dict[str, Any]]]
        The network configuration file
    latent_dim : int
        The new latent dimension to set
    """
    i: int = 0
    layer: dict[str, Any]

    for i, layer in enumerate(config['layers'][::-1]):
        if layer['type'] == 'Checkpoint':
            break

    config['layers'][-i - 2]['features'] = latent_dim


def net_accuracy(kwargs: Any) -> None:
    """
    Calculate the accuracy of the network on the test set and stores it in data['accuracy'] from
    locals.

    Parameters
    ----------
    **kwargs :
        excluded_columns : list[str]
            Columns to exclude from the dataset
        loaders : list[DataLoader]
            The data loaders for the dataset
        data : dict[str, Any]
            Dictionary to store the accuracy in
        net : BaseNetwork
            The network to evaluate
    """
    data: dict[str, Any]
    net: BaseNetwork = kwargs['net']

    if 'accuracy' not in kwargs['excluded_columns']:
        kwargs['excluded_columns'].append('accuracy')

    data = net.predict(kwargs['loaders'][-1])
    kwargs['data']['accuracy'] = np.count_nonzero(
        data['preds'] == data['targets'].squeeze()
    ) / len(data['targets'])


def generate_test_configs(
        repeats: int,
        net_name: str,
        data_dir: str,
        latent_dims: list[int],
        sim_sets: list[list[str]],
        cumulative: bool = False,
        description: str = '',
        hyperparams: dict[str, Any] | None = None) -> list[TestConfig]:
    """
    Generate a list of test configurations for different latent dimensions and simulation sets.

    Parameters
    ----------
    repeats : int
        Number of times to repeat each test configuration
    net_name : str
        Network configuration file name
    data_dir : str
        Path to the data directory
    latent_dims : list[int]
        List of latent dimensions to test
    sim_sets : list[list[str]]
        List of simulation sets to test
    cumulative : bool, default = False
        If simulations should be cumulatively added
    description : str, default = ''
        Description of the tests
    hyperparams : dict[str, Any] | None, default = None
        Optional hyperparameters for the tests

    Returns
    -------
    list[TestConfig]
        List of test configurations
    """
    tests = []
    current_known: ndarray = np.array([])

    for i, sims in enumerate(sim_sets):
        if cumulative:
            current_known = update_sims(sims, current_known)
        else:
            current_known = np.array(sims)

        for j, latent_dim in enumerate(latent_dims):
            for k in range(repeats):
                tests.append(
                    TestConfig(
                        name=f'{i}.{j}.{k}',
                        net_name=net_name,
                        description=(f'{description}\n' if description else '') +
                        f'Latent Dim: {latent_dim}\nSims: {current_known}',
                        dataset_args={
                            'data_dir': data_dir,
                            'sims': current_known.tolist(),
                            'unknown_sims': [],
                        },
                        network_mod_params={'latent_dim': latent_dim},
                        hyperparams=hyperparams,
                        network_mod_fn=change_latent_dim,
                        post_train_fn=net_accuracy,
                    )
                )
    return tests


def main(config_path: str = '../config.yaml') -> None:
    """
    Main function to run the latent dimension tests.

    Parameters
    ----------
    config_path : str, default = '../config.yaml'
        Path to the configuration file
    """
    x_rays: bool = False
    skip: int = 2
    repeats: int = 5
    epochs: int = 150
    channels: int
    save: int | str
    load: int | str
    description: str
    latent_dims: list[int] = np.concat((
        np.arange(1, 10),
        np.arange(10, 22, 2),
    )).tolist() + [50, 100, 1000]
    sim_sets: list[list[str]] = [
        ['bahamas_cdm', 'bahamas_0.1', 'bahamas_0.3', 'bahamas_1'],
        ['bahamas_cdm_low', 'bahamas_cdm_hi'],
        ['darkskies_cdm', 'darkskies_0.1', 'darkskies_0.2'],
        ['flamingo', 'flamingo_low', 'flamingo_hi'],
    ][:-1]
    config: dict[str, Any]
    hyperparams: dict[str, Any] = {'epochs': epochs}
    test: TestConfig

    if x_rays:
        save = 'test_latent_1'
        load = save
        channels = 2
        description = 'X-Rays'
    else:
        save = 'test_latent_2'
        load = save
        channels = 1
        description = 'No X-Rays'

    _, config = open_config('main', config_path)
    config['training']['seed'] = -1
    config['training']['image-channels'] = channels
    config['training']['cdm-sigma'] = 1e-2
    tests = generate_test_configs(
        repeats,
        'network_v10',
        config['data']['data-dir'],
        latent_dims,
        sim_sets,
        cumulative=True,
        description=description,
        hyperparams=hyperparams,
    )
    sub_tests = [
        test for test in tests
        if test.network_mod_params['latent_dim'] in (latent_dims[::skip] + [2])
    ]

    run_tests(
        save,
        load,
        os.path.join(config['data']['data-dir'], f'{save}.pkl'),
        tests,
        config=config,
    )


if __name__ == '__main__':
    main()
