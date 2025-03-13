"""
Misc functions used elsewhere
"""
import os
from typing import Any
from argparse import ArgumentParser

import yaml


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

    with open(config_path, 'rb') as file:
        config = yaml.safe_load(file)[key]

    return config_path, config
