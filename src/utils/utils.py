"""
Misc functions used elsewhere
"""
import os
from typing import Any
from argparse import ArgumentParser

import yaml
import wandb  # pylint: disable=wrong-import-order
import numpy as np
from numpy import ndarray
from scipy.stats import gaussian_kde


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _interactive_check() -> bool:
    """
    Checks if the launch environment is interactive or not

    Returns
    -------
    bool
        If environment is interactive
    """
    if os.getenv('PYCHARM_HOSTED'):
        return True

    try:
        if get_ipython().__class__.__name__:
            return True
    except NameError:
        return False

    return False


def dict_list_append(
        dict1: dict[str, list[float | None] | list[ndarray]],
        dict2: dict[str, float | list[float] | list[ndarray] | ndarray],
) -> dict[str, list[float | None] | list[ndarray]]:
    """
    Merges two dictionaries of lists

    Parameters
    ----------
    dict1 : dict[str, list[float | None] | list[ndarray]]
        Primary dict to merge secondary dict into, can be empty
    dict2 : dict[str, float | list[float] | list[ndarray] | ndarray]
        Secondary dict to merge into primary dict, requires at least one element

    Returns
    -------
    dict[str, list[float | None] | list[ndarray]]
        First dict with second dict merged into it
    """
    dict1_len: int = 0
    dict2_len: int = 1
    key: str

    # If primary dict is not empty, find the length of a list in the dictionary
    if len(dict1.keys()) > 0:
        dict1_len = len(dict1[list(dict1.keys())[0]])

    # If the secondary dict contains a list of items, find the length of the lists
    if isinstance(dict2[list(dict2.keys())[0]], list):
        dict2_len = len(dict2[list(dict2.keys())[0]])

    # Merge two dictionaries
    for key in np.unique(list(dict1.keys()) + list(dict2.keys())):
        key = str(key)

        if key == 'galaxy_catalogues':
            dict2[key] = np.array(dict2[key], dtype=object)

        # If the secondary dict has a key not in the primary, pad with Nones
        if key not in dict1 and np.ndim(dict2[key]) > 0 and isinstance(dict2[key][0], ndarray):
            dict1[key] = [np.array([None] * len(dict2[key][0]))] * dict1_len
        elif key not in dict1:
            dict1[key] = [None] * dict1_len

        # If the primary dict has a key not in the secondary dict, pad with Nones, else merge dicts
        if key not in dict2 and isinstance(dict1[key][0], ndarray):
            dict1[key].extend([np.array([None] * len(dict1[key][0]))] * dict2_len)
        elif key not in dict2:
            dict1[key].extend([None] * dict2_len)
        # elif np.ndim(dict2[key]) > 0:
        #     dict1[key].extend(dict2[key])
        else:
            dict1[key].append(dict2[key])
    return dict1


def list_dict_convert(
        data: list[dict[str, float | ndarray]],
) -> dict[str, list[float | None] | list[ndarray]]:
    """
    Converts a list of dictionaries to a dictionary of lists

    Parameters
    ----------
    data : list[dict[str, float | ndarray]]
        List of dictionaries to convert

    Returns
    -------
    dict[str, list[float | None] | list[ndarray]]
        Dictionary of lists
    """
    value: dict[str, float | ndarray]
    new_data: dict[str, list[float] | list[ndarray]] = {}

    for value in data:
        dict_list_append(new_data, value)
    return new_data


def dict_list_convert(data: dict[str, list[Any] | ndarray]) -> list[dict[str, Any]]:
    """
    Converts a dictionary of lists to a list of dictionaries.

    Parameters
    ----------
    data : dict[str, list[Any] | ndarray]
        Dictionary of lists to convert

    Returns
    -------
    list[dict[str, Any]]
        List of dictionaries
    """
    return [{key: data[key][i] for key in data} for i in range(len(list(data.values())[0]))]


def open_config(
        key: str,
        config_path: str,
        parser: ArgumentParser | None = None) -> tuple[str, dict[str, Any]]:
    """
    Opens the configuration file from either the provided path or through command line argument

    Parameters
    ----------
    key : str
        Key of the configuration file
    config_path : str
        Default path to the configuration file
    parser : ArgumentParser | None, default = None
        Parser if arguments other than config path are required

    Returns
    -------
    tuple[str, dict[str, Any]]
        Configuration path and configuration file dictionary
    """
    if not _interactive_check():
        if not parser:
            parser = ArgumentParser()

        parser.add_argument(
            '--config_path',
            default=config_path,
            help='Path to the configuration file',
            required=False,
        )
        args = parser.parse_args()
        config_path = args.config_path

    config_path += '' if '.yaml' in config_path else '.yaml'

    with open(os.path.join(ROOT, config_path), 'rb') as file:
        config = yaml.safe_load(file)[key]

    return config_path, config


def overlap(data_1: np.ndarray, data_2: np.ndarray, bins: int = 100) -> float:
    """
    Calculates the overlap between two datasets by using a Gaussian kernel to approximate the
    distribution, then integrates the overlap using the trapezoidal rule

    Parameters
    ----------
    data_1 : ndarray
        First dataset of shape (N), where N are the number of points
    data_2 : ndarray
        Second dataset of shape (M), where M are the number of points
    bins : int, default = 100
        Number of bins to sample from the Gaussian distribution approximation

    Returns
    -------
    float
        Overlap fraction
    """
    grid = np.linspace(min(data_1.min(), data_2.min()), max(data_1.max(), data_2.max()), bins)
    kde_1 = gaussian_kde(data_1)
    kde_2 = gaussian_kde(data_2)

    pdf_1 = kde_1(grid)
    pdf_2 = kde_2(grid)
    return np.trapezoid(np.minimum(pdf_1, pdf_2), grid)


def run_exists(entity: str, project: str, run_id: str) -> bool:
    """
    Checks if a run exists in Weights & Biases.

    Parameters
    ----------
    entity : str
        W&B entity name
    project : str
        W&B project name
    run_id : str
        W&B run ID

    Returns
    -------
    bool
        If the run exists
    """
    try:
        wandb.Api().run(os.path.join(entity, project, run_id))
        return True
    except wandb.errors.CommError:
        return False


def wandb_config(
        epochs: int,
        name: str,
        group: str,
        config: dict[str, Any]) -> dict[str, str | dict[str, Any]]:
    """
    Creates a Weights & Biases configuration dictionary.

    Parameters
    ----------
    epochs : int
        Number of training epochs
    name : str
        Unique name of the run
    group : str
        Group name of the run
    config : dict[str, Any]
        Configuration dictionary

    Returns
    -------
    dict[str, str | dict[str, Any]]
        Weights & Biases configuration dictionary
    """
    return {
        'entity': 'davidharvey1986-epfl',
        'project': 'Bayesian-DARKSKIES',
        # 'id': f'{net.save_path.split('/')[-1].replace('.pth', '')}-{generate_id(4)}',
        'id': name,
        'name': name,
        'group': group,
        'config': {'epochs': epochs} | config,
        # 'config': {
        #     'epochs': epochs,
        #     'batch_size': config['training']['batch-size'],
        #     'steps': net.get_steps(),
        #     'learning_rate': config['training']['learning-rate'],
        #     'max_learning_rate': config['training']['max-learning-rate'],
        #     'validation_fraction': config['training']['validation-fraction'],
        #     'Classification Weight': net.class_loss,
        #     'CCLP Weight': net.cluster_loss,
        #     'Distance Weight': net.distance_loss,
        #     'known': known,
        #     'unknown': unknown,
        #     'description': net.description,
        #     'optimiser': net.optimiser.__class__.__name__,
        #     'scheduler': net.scheduler.__class__.__name__,
        # },
    }
