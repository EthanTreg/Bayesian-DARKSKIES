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

    with open(config_path, 'rb') as file:
        config = yaml.safe_load(file)[key]

    return config_path, config


def progress_bar(i: int, total: int, text: str = '', **kwargs: Any) -> None:
    """
    Terminal progress bar

    Parameters
    ----------
    i : int
        Current progress
    total : int
        Completion number
    text : str, default = ''
        Optional text to place at the end of the progress bar

    **kwargs
        Optional keyword arguments to pass to print
    """
    filled: int
    length: int = 50
    percent: float
    bar_fill: str
    i += 1

    filled = int(i * length / total)
    percent = i * 100 / total
    bar_fill = '█' * filled + '-' * (length - filled)
    print(f'\rProgress: |{bar_fill}| {int(percent)}%\t{text}\t', end='', **kwargs)

    if i == total:
        print()
