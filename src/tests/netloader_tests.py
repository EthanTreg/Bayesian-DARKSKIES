"""
Functions to run tests on neural networks using the PyTorch-Network-Loader package.
"""
import os
import json
import pickle
from warnings import warn
from dataclasses import dataclass
from typing import Callable, Any, TextIO

import numpy as np
import pandas as pd
from netloader.data import loader_init
from netloader.networks import BaseNetwork
from netloader.utils.utils import save_name
from torch.utils.data import DataLoader

from src.main import net_init
from src.utils.data import DarkDataset
from src.utils.utils import open_config, ROOT


@dataclass
class TestConfig:
    # pylint: disable=line-too-long
    """
    Configuration for a test to be run on the neural network.

    Attributes
    ----------
    name : str
        Name of the test
    net_name : str
        Name of the network configuration file to use
    description : str
        Description of the test
    dataset_args : dict[str, Any]
        Keyword arguments for the dataset initialization
    hyperparams : dict[str, Any] | None, default = None
        Optional hyperparameters for the test
    pre_train_fn_params : dict[str, Any] | None, default = None
        Keyword arguments for the custom function before network training
    post_train_fn_params : dict[str, Any] | None, default = None
        Keyword arguments for the custom function after network training
    network_mod_params : dict[str, Any] | None, default = None
        Keyword arguments for the network modification function
    network_mod_fn : Callable[[dict[str, dict[str, Any] | list[dict[str, Any]]], Any], None] | None, default = None
        Function to modify the network configuration before training
    pre_train_fn : Callable[[Any], None] | None, default = None
        Function to run custom logic before training, receives locals() as argument
    post_train_fn : Callable[[Any], None] | None, default = None
        Function to run custom logic after training, receives locals() as argument
    """
    # pylint: enable=line-too-long
    name: str
    net_name: str
    description: str
    dataset_args: dict[str, Any]
    hyperparams: dict[str, Any] | None = None
    custom_fn_params: dict[str, Any] | None = None
    network_mod_params: dict[str, Any] | None = None
    pre_train_fn_params: dict[str, Any] | None = None
    post_train_fn_params: dict[str, Any] | None = None
    network_mod_fn: Callable[
                        [dict[str, dict[str, Any] | list[dict[str, Any]]], Any],
                        None,
                    ] | None = None
    custom_fn: Callable[[...], None] | None = None
    pre_train_fn: Callable[[...], None] | None = None
    post_train_fn: Callable[[...], None] | None = None

    def __post_init__(self) -> None:
        if self.custom_fn is not None:
            warn(
                'custom_fn is deprecated, use pre_train_fn and post_train_fn instead',
                DeprecationWarning,
                stacklevel=2,
            )


def mod_network(nets_dir: str, test: TestConfig, suffix: str = 'temp') -> str:
    """
    Modify the network JSON file according to the test configuration and save as a temp file.

    Parameters
    ----------
    nets_dir : str
        Path to the network configurations directory
    test : TestConfig
        The test configuration containing the network modification parameters
    suffix : str, default = 'temp'
        Suffix to append to the new temporary file name, if empty, 'temp' will be used

    Returns
    -------
    str
        Path to the new temporary JSON file with updated latent dimension
    """
    new_name: str
    config: dict[str, dict[str, Any] | list[dict[str, Any]]]
    file: TextIO
    suffix = suffix or 'temp'

    if '.json' in test.net_name:
        test.net_name.replace('.json', '')

    if not test.network_mod_fn:
        return test.net_name

    new_name = f'{test.net_name}_{suffix}'

    with open(os.path.join(ROOT, nets_dir, test.net_name) + '.json', 'r', encoding='utf-8') as file:
        config = json.load(file)

    test.network_mod_fn(config, **test.network_mod_params or {})

    with open(os.path.join(ROOT, nets_dir, new_name) + '.json', 'w', encoding='utf-8') as file:
        json.dump(config, file)
    return new_name


def gen_indexes(data: pd.DataFrame, excluded_columns: list[str] | None = None) -> None:
    """
    Generates unique indexes for a data frame.

    Parameters
    ----------
    data : pd.DataFrame
        Data frame to generate indexes for
    excluded_columns : list[str] | None, default = None
        Columns to exclude from index generation
    """
    key: str
    idxs: np.ndarray
    vals: np.ndarray

    if len(data) < 2:
        return

    # Add index columns for columns with unique values
    for key in data.keys():
        if key in (excluded_columns or []):
            continue

        try:
            idxs = pd.factorize(data[key])[0]
        except TypeError:
            idxs = pd.factorize(data[key].apply(tuple))[0]

        if 1 < len(np.unique(idxs)) < len(data):
            data[f'{key}_idx'] = idxs
            data.set_index(f'{key}_idx', inplace=True, append=True)

    if isinstance(data.index, pd.MultiIndex):
        vals = np.array(data.index.values.tolist()).swapaxes(0, 1)
        idxs = np.sort(np.unique(vals, axis=0, return_index=True)[1])
        data.index = pd.MultiIndex.from_arrays(vals[idxs], names=np.array(data.index.names)[idxs])


def run_tests(
        save: int | str,
        load: int | str,
        results_path: str,
        tests: list[TestConfig],
        config: str | dict[str, Any] = '../config.yaml') -> None:
    """
    Run a series of tests on the neural network using the provided configurations.

    Parameters
    ----------
    save : int | str
        File name to save the results, or 0 to not save
    load : int | str
        File name to load the networks from, or 0 to not load
    results_path : str
        Path to save the results DataFrame
    tests : list[TestConfig]
        Test configurations
    config : str | dict[str, Any], default = '../config.yaml'
        Configuration file path or dictionary
    """
    i: int
    val_frac: float
    nets_dir: str
    net_name: str
    loaders: tuple[DataLoader, ...]
    excluded_columns: list[str] = ['net_path', 'description', 'losses']
    data: dict[str, Any] = {}
    dataset_args: dict[str, Any] = {}
    results: pd.DataFrame = pd.DataFrame([])
    net: BaseNetwork
    dataset: DarkDataset
    test: TestConfig

    if isinstance(config, str):
        _, config = open_config('main', config)

    nets_dir = str(os.path.join(ROOT, config['data']['network-configs-directory']))

    for i, test in enumerate(tests):
        print(f'\nRunning test ({i + 1}/{len(tests)}): {test.description}', flush=True)

        # 1. Prepare network JSON
        net_name = mod_network(nets_dir, test, suffix=f'temp_{save}')

        # 2. Prepare config
        config['training'].update(test.hyperparams or {})
        config['training']['network-name'] = net_name
        config['training']['network-save'] = f'{test.name}' if save else 0
        config['training']['network-load'] = f'{test.name}'
        config['training']['description'] = test.description
        val_frac = config['training']['validation-fraction']

        if not load or not os.path.exists(save_name(
            config['training']['network-load'],
            str(os.path.join(ROOT, config['output']['network-states-directory'])),
            config['training']['network-name'],
        )):
            config['training']['network-load'] = 0

        # 3. Prepare or reuse dataset
        if not test.dataset_args == dataset_args:
            dataset = DarkDataset(**test.dataset_args)
            dataset.low_dim = dataset.unique_labels(dataset.low_dim, dataset.extra['sims'])
            dataset_args = test.dataset_args

        # 4. Continue as before, using the cached dataset
        net = net_init(dataset, config=config)
        loaders = loader_init(
            dataset,
            batch_size=config['training']['batch-size'],
            ratios=(1 - val_frac, val_frac) if net.idxs is None else (1,),
            idxs=None if net.idxs is None else dataset.idxs[np.isin(
                dataset.extra['ids'],
                net.idxs,
            )],
        )
        net.idxs = dataset.extra['ids'].iloc[loaders[0].dataset.indices] if net.idxs is None else \
            net.idxs

        # 4. Custom user logic (optional)
        if test.custom_fn:
            test.custom_fn(locals(), **test.custom_fn_params or {})

        if test.pre_train_fn:
            test.pre_train_fn(locals(), **test.pre_train_fn_params or {})

        # 5. Train and evaluate
        if net.get_epochs() < config['training']['epochs']:
            net.training(config['training']['epochs'], loaders)
            net.save()

        if test.post_train_fn:
            test.post_train_fn(locals(), **test.post_train_fn_params or {})

        dataset.high_dim = net.transforms['inputs'](dataset.high_dim, back=True)
        dataset.low_dim = net.transforms['targets'](dataset.low_dim, back=True)
        results = pd.concat((None if results.empty else results, pd.DataFrame([{
            'net_path': net.save_path,
            'description': test.description,
            'losses': np.array(net.losses),
            'network_mod_fn': test.network_mod_fn.__name__ if test.network_mod_fn else None,
            'custom_fn': test.custom_fn.__name__ if test.custom_fn else None,
            **(test.dataset_args or {}),
            **(test.hyperparams or {}),
            **(test.network_mod_params or {}),
            **(test.custom_fn_params or {}),
            **(test.pre_train_fn_params or {}),
            **(test.post_train_fn_params or {}),
            **data,
        }])), ignore_index=True)
        gen_indexes(results, excluded_columns=excluded_columns)

        if test.network_mod_fn:
            os.remove(os.path.join(nets_dir, net_name) + '.json')

    # 6. Save results
    if save:
        with open(os.path.join(ROOT, results_path), 'wb') as file:
            pickle.dump(results, file)
