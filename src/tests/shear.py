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


def set_net_hyperparams(
        kwargs: dict[str, Any],
        *,
        steps: int,
        class_loss: float,
        cluster_loss: float,
        distance_loss: float) -> None:
    """
    Set network hyperparameters for training.

    Parameters
    ----------
    kwargs : Any
        Local keyword arguments containing the network instance
    steps : int
        Number of compact clustering Monte Carlo steps
    class_loss : float
        Weighting for the classification loss
    cluster_loss : float
        Weighting for the clustering loss
    distance_loss : float
        Weighting for the distance loss
    """
    net = kwargs['net']
    net.steps = steps
    net.class_loss = class_loss
    net.cluster_loss = cluster_loss
    net.distance_loss = distance_loss


def generate_test_configs(
        repeats: int,
        net_name: str,
        data_dir: str,
        sim_sets: list[list[str]],
        unknown_sets: list[list[str]],
        cumulative: bool = False,
        description: str = '',
        cdm_sigmas: list[float] | None = None,
        custom_fn_params: list[dict[str, int | float]] | None = None,
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
    cdm_sigmas : list[float] | None, default = [1e-2]
        Effective cross-section for CDM simulations
    description : str, default = ''
        Description of the tests
    custom_fn_params : list[dict[str, int | float]] | None, default = {}
        List of custom function parameters for each test
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

        for j, cdm_sigma in enumerate(cdm_sigmas or [1e-2]):
            for k, custom_fn_param in enumerate(custom_fn_params or [{}]):
                for l in range(repeats):
                    tests.append(
                        TestConfig(
                            name=f'{i}.{j}.{k}.{l}',
                            net_name=net_name,
                            description=f'{description}\n' if description else ''
                            f'Sims: {current_known}\nUnknown Sims: {current_unknown}\n'
                            f'Params: {custom_fn_param}\nCDM Sigma: {cdm_sigma}',
                            dataset_args={
                                'data_dir': data_dir,
                                'sims': current_known.tolist(),
                                'unknown_sims': current_unknown.tolist(),
                                'cdm_sigma': cdm_sigma,
                            },
                            hyperparams=hyperparams,
                            custom_fn_params=custom_fn_param,
                            custom_fn=set_net_hyperparams,
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
    repeats: int = 5
    epochs: int = 150
    save: int | str = 'shear_9'
    load: int | str = save
    cdm_sigmas: list[float]
    sim_sets: list[list[str]]
    unknown_sets: list[list[str]]
    custom_fn_params: list[dict[str, int | float]]
    config: dict[str, Any]
    hyperparams: dict[str, Any] = {'epochs': epochs}

    # shear_8 & shear_9
    sim_sets = [['bahamas_cdm_shear', 'bahamas_0.1_shear', 'bahamas_1_shear']]
    unknown_sets = [['bahamas_0.3_shear']]

    # shear_8
    # cdm_sigmas = [1e-3, 3e-3, 1e-2]
    # custom_fn_params = [
    #     {'steps': 3, 'class_loss': 1, 'cluster_loss': 1, 'distance_loss': 1},
    #     {'steps': 3, 'class_loss': 1, 'cluster_loss': 1, 'distance_loss': 0.5},
    #     {'steps': 3, 'class_loss': 1, 'cluster_loss': 0.5, 'distance_loss': 1},
    #     {'steps': 3, 'class_loss': 1, 'cluster_loss': 0.5, 'distance_loss': 0.5},
    #     {'steps': 3, 'class_loss': 0.5, 'cluster_loss': 1, 'distance_loss': 1},
    #     {'steps': 3, 'class_loss': 0.5, 'cluster_loss': 1, 'distance_loss': 0.5},
    #     {'steps': 3, 'class_loss': 0.5, 'cluster_loss': 0.5, 'distance_loss': 1},
    # ]

    # shear_9
    cdm_sigmas = [1e-3, 1e-2]
    custom_fn_params = [
        {'steps': 3, 'class_loss': 1, 'cluster_loss': 1, 'distance_loss': 0.5},
        {'steps': 5, 'class_loss': 1, 'cluster_loss': 1, 'distance_loss': 0.5},
        {'steps': 7, 'class_loss': 1, 'cluster_loss': 1, 'distance_loss': 0.5},
        {'steps': 9, 'class_loss': 1, 'cluster_loss': 1, 'distance_loss': 0.5},
    ]

    _, config = open_config('main', config_path)
    tests = generate_test_configs(
        repeats,
        'network_v17',
        config['data']['data-dir'],
        sim_sets,
        unknown_sets,
        cumulative=False,
        cdm_sigmas=cdm_sigmas,
        custom_fn_params=custom_fn_params,
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
