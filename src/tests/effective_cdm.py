"""
Test script for varying CDM effective cross-section and simulations sets in a neural network.
"""
import os
from typing import Any
from itertools import repeat

import numpy as np
from numpy import ndarray

from src.batch_train import update_sims
from src.utils.utils import open_config
from src.tests.netloader_tests import TestConfig, run_tests


def generate_test_configs(
        repeats: int,
        net_name: str,
        data_dir: str,
        sim_sets: list[list[str]],
        unknown_sets: list[list[str]],
        cumulative: bool = False,
        description: str = '',
        cdm_sigmas: list[list[float]] | list[float] | repeat | None = None,
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
    sim_sets : list[list[str]]
        List of simulation sets to test
    unknown_sets : list[list[str]]
        List of unknown simulation sets to test
    cumulative : bool, default = False
        If simulations should be cumulatively added
    cdm_sigmas : list[list[float]] | list[float] | repeat | None, default = repeat([1e-2])
        Effective cross-section for CDM simulations
    description : str, default = ''
        Description of the tests
    hyperparams : dict[str, Any] | None, default = None
        Optional hyperparameters for the tests

    Returns
    -------
    list[TestConfig]
        List of test configurations
    """
    i: int
    j: int
    k: int
    cdm_sigma: float
    cdm_set: list[float]
    sims: list[str]
    unknown_sims: list[str]
    tests: list[TestConfig] = []
    current_known: ndarray = np.array([])
    current_unknown: ndarray = np.array([])

    if not cdm_sigmas:
        cdm_sigmas = repeat([1e-2])
    elif not isinstance(cdm_sigmas, repeat):
        cdm_sigmas = cdm_sigmas if isinstance(cdm_sigmas[0], list) else repeat(cdm_sigmas)

    for i, (sims, unknown_sims, cdm_set) in enumerate(zip(
            sim_sets,
            unknown_sets,
            cdm_sigmas)):
        if cumulative:
            current_known = update_sims(sims, current_known)
            current_unknown = update_sims(unknown_sims, current_unknown)
        else:
            current_known = np.array(sims)
            current_unknown = np.array(unknown_sims)

        for j, cdm_sigma in enumerate(cdm_set):
            for k in range(repeats):
                tests.append(
                    TestConfig(
                        name=f'{i}.{j}.{k}',
                        net_name=net_name,
                        description=f'{description}\n' if description else ''
                        f'Sims: {current_known}\nUnknown Sims: {current_unknown}\n'
                        f'CDM Sigma: {cdm_sigma}',
                        dataset_args={
                            'data_dir': data_dir,
                            'sims': current_known.tolist(),
                            'unknown_sims': current_unknown.tolist(),
                            'cdm_sigma': cdm_sigma,
                        },
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
    save: int | str = 'cdm_2'
    load: int | str = save
    cdm_sigmas: list[list[float]] = [[1e-3, 3e-3, 1e-2, 3e-2, 1e-1], [1e-3]]
    sim_sets: list[list[str]] = [
        ['bahamas_cdm', 'bahamas_0.3', 'bahamas_1'],
        ['bahamas_0.3', 'bahamas_1'],
    ]
    unknown_sets: list[list[str]] = [['bahamas_0.1'], ['bahamas_cdm', 'bahamas_0.1']]
    config: dict[str, Any]
    hyperparams: dict[str, Any] = {'epochs': epochs}

    _, config = open_config('main', config_path)
    config['training']['image-channels'] = 1
    tests = generate_test_configs(
        repeats,
        'network_v10',
        config['data']['data-dir'],
        sim_sets,
        unknown_sets,
        cumulative=False,
        cdm_sigmas=cdm_sigmas,
        hyperparams=hyperparams,
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
