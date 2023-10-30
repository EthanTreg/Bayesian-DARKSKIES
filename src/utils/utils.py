"""
Misc functions used elsewhere
"""
import os
from argparse import ArgumentParser

import yaml
import torch


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
