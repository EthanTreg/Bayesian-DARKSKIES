"""
Test script for varying latent dimensions and simulations sets in a neural network.
"""
import os
from typing import Any

import numpy as np
from numpy import ndarray

from src.batch_train import update_sims
from src.utils.utils import open_config
from src.tests.netloader_tests import TestConfig, run_tests


def generate_test_configs(
        repeats: int,
        net_name: str,
        data_dir: str,
        mix_up: list[bool],
        sim_sets: list[list[str]],
        unknown_sets: list[list[str]],
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
    mix_up: list[bool]
        List of booleans indicating whether to mix up the simulations
    sim_sets : list[list[str]]
        List of simulation sets to test
    unknown_sets : list[list[str]]
        List of unknown simulation sets to test
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
    current_unknown: ndarray = np.array([])

    for i, (sims, unknown_sims) in enumerate(zip(sim_sets, unknown_sets)):
        if cumulative:
            current_known = update_sims(sims, current_known)
            current_unknown = update_sims(unknown_sims, current_unknown)
        else:
            current_known = np.array(sims)
            current_unknown = np.array(unknown_sims)

        for j, mix in enumerate(mix_up):
            for k in range(repeats):
                tests.append(
                    TestConfig(
                        f'{i}.{j}.{k}',
                        f'{description}\n' if description else ''
                        f'Mix Up: {mix}\nSims: {current_known}\nUnknown Sims: {current_unknown}',
                        {
                            'data_dir': data_dir,
                            'sims': current_known.tolist(),
                            'unknown_sims': current_unknown.tolist(),
                            'mix': mix,
                        },
                        net_name,
                        hyperparams=hyperparams,
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
    repeats: int = 3
    epochs: int = 150
    save: int | str = 'mix_1'
    load: int | str = save
    mix_up: list[bool] = [True, False]
    sim_sets: list[list[str]] = [
        ['bahamas_cdm', 'bahamas_0.1', 'bahamas_1'],
        ['bahamas_cdm', 'bahamas_0.3', 'bahamas_1'],
    ]
    unknown_sets: list[list[str]] = [['bahamas_0.3'], ['bahamas_0.1']]
    config: dict[str, Any]
    hyperparams: dict[str, Any] = {'epochs': epochs}

    _, config = open_config('main', config_path)
    tests = generate_test_configs(
        repeats,
        'network_v10',
        config['data']['data-dir'],
        mix_up,
        sim_sets,
        unknown_sets,
        hyperparams=hyperparams,
        cumulative=True,
    )
    run_tests(
        save,
        load,
        os.path.join(config['data']['data-dir'], f'{save}.pkl'),
        tests,
        config=config,
    )


if __name__ == '__main__':
    main()
