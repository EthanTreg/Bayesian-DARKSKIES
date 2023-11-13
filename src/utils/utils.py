"""
Misc functions used elsewhere
"""
import os
from argparse import ArgumentParser

import yaml
import torch
import numpy as np
from numpy import ndarray


def _interactive_check() -> bool:
    """
    Checks if the launch environment is interactive or not

    Returns
    -------
    boolean
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


def get_device() -> tuple[dict, torch.device]:
    """
    Gets the device for PyTorch to use

    Returns
    -------
    tuple[dictionary, device]
        Arguments for the PyTorch DataLoader to use when loading data into memory and PyTorch device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    return kwargs, device


def save_name(num: int, states_dir: str, name: str) -> str:
    """
    Standardises the network save file naming

    Parameters
    ----------
    num : integer
        File number
    states_dir : string
        Directory of network saves
    name : string
        Name of the network

    Returns
    -------
    string
        Path to the network save file
    """
    return f'{states_dir}{name}_{num}.pth'


def name_sort(
        names: list[ndarray, ndarray],
        data: list[ndarray, ndarray],
        shuffle: bool = True) -> tuple[list[ndarray, ndarray], list[ndarray, ndarray]]:
    """
    Sorts names and data so that two arrays contain the same names

    Parameters
    ----------
    names : list[ndarray, ndarray]
        Name arrays to sort
    data : list[ndarray, ndarray]
        Data arrays to sort from corresponding name arrays
    shuffle : boolean, default = True
        If name and data arrays should be shuffled

    Returns
    -------
    tuple[list[ndarray, ndarray], list[ndarray, ndarray]]
        Sorted name and data arrays
    """
    # Sort for longest dataset first
    sort_idx = np.argsort([datum.shape[0] for datum in data])[::-1]
    data = [data[i] for i in sort_idx]
    names = [names[i] for i in sort_idx]

    # Sort target spectra by name
    name_sort_idx = np.argsort(names[0])
    names[0] = names[0][name_sort_idx]
    data[0] = data[0][name_sort_idx]

    # Filter target parameters using spectra that was predicted and log parameters
    target_idx = np.searchsorted(names[0], names[1])
    names[0] = names[0][target_idx]
    data[0] = data[0][target_idx]

    # Shuffle params
    if shuffle:
        shuffle_idx = np.random.permutation(data[0].shape[0])
        names[0] = names[0][shuffle_idx]
        names[1] = names[1][shuffle_idx]
        data[0] = data[0][shuffle_idx]
        data[1] = data[1][shuffle_idx]

    data = [data[i] for i in sort_idx]
    names = [names[i] for i in sort_idx]

    return names, data


def open_config(key: str, config_path: str, parser: ArgumentParser = None) -> tuple[str, dict]:
    """
    Opens the configuration file from either the provided path or through command line argument

    Parameters
    ----------
    key : string
        Key of the configuration file
    config_path : string
        Default path to the configuration file
    parser : ArgumentParser, default = None
        Parser if arguments other than config path are required

    Returns
    -------
    tuple[string, dictionary]
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

    with open(config_path, 'rb') as file:
        config = yaml.safe_load(file)[key]

    return config_path, config
