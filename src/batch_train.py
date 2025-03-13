"""
Trains a network multiple times for different datasets
"""
import os
import pickle
from typing import Any

import numpy as np
from numpy import ndarray
from netloader.networks import BaseNetwork
from netloader.utils.utils import save_name
from torch.utils.data import DataLoader, Dataset

from src.utils import analysis
from src.main import init, net_init
from src.utils.utils import open_config


def main(config_path: str = '../config.yaml'):
    """
    Main function for batch training of the network

    Parameters
    ----------
    config_path : string, default = '../config.yaml'
        Path to the configuration file
    """
    cumulative: bool
    unknown_cumulative: bool
    i: int
    j: int
    epochs: int
    repeats: int
    save_num: int
    key: str
    name: str
    net_name: str
    states_dir: str
    current_name: str = ''
    sims: list[str]
    unknown_sims: list[str]
    current_known: list[str] = []
    current_unknown: list[str] = []
    known: list[list[str]]
    unknown: list[list[str]]
    loaders: tuple[DataLoader, DataLoader]
    config: dict[str, Any]
    batch_config: dict[str, Any]
    results: dict[int, dict[str, str | list[ndarray] | ndarray]] = {}
    meds: ndarray
    stes: ndarray
    means: ndarray
    dataset: Dataset
    net: BaseNetwork

    _, batch_config = open_config('batch-train', config_path)
    _, config = open_config('main', config_path)
    _, batch_config['training'] = open_config(
        'training',
        os.path.join(batch_config['data']['config-dir'], batch_config['data']['config']),
    )

    for key, value in batch_config.items():
        config[key] |= value

    cumulative = config['training']['cumulative']
    unknown_cumulative = config['training']['unknown-cumulative']
    epochs = config['training']['epochs']
    repeats = config['training']['repeats']
    save_num = config['training']['batch-save']
    load_num = config['training']['batch-load']
    net_name = config['training']['network-name']
    description = config['training']['description']
    names = config['training']['test-names']
    known = config['training']['known-simulations']
    unknown = config['training']['unknown-simulations']
    states_dir = config['output']['network-states-directory']

    if len(unknown) == 1:
        unknown *= len(known)

    if len(known) == 1:
        known *= len(unknown)

    if len(known) != len(unknown):
        raise ValueError(f'Number of known simulation tests ({len(known)}) does not equal the '
                         f'number of unknown simulation tests ({len(unknown)})')

    for i, (sims, unknown_sims, name) in enumerate(zip(known, unknown, names)):
        if cumulative:
            current_known += sims
            current_name = f'{current_name} + {name}' if current_name else name
        else:
            current_known = sims
            current_name = name

        if unknown_cumulative:
            current_unknown += unknown_sims
        else:
            current_unknown = unknown_sims

        config['training']['description'] = (f'{current_name}'
                                             f"{', ' + description if description else ''}")
        results[i] = {
            'meds': [],
            'means': [],
            'stes': [],
            'log_meds': [],
            'log_means': [],
            'log_stes': [],
            'targets': [],
            'nets': [],
            'description': config['training']['description'],
            'sims': current_known,
            'unknown_sims': current_unknown,
        }

        # Initialise datasets & loaders
        config['training']['network-load'] = 0
        config['training']['network-save'] = 0
        loaders, net, dataset = init(current_known, config, unknown=current_unknown)
        # loaders, net, dataset = init(current_known + current_unknown, config) # For Encoders

        if (np.char.find(current_unknown, 'dmo') != -1).any():
            dataset.images = dataset.images[:, :1]

        for j in range(repeats):
            config['training']['network-save'] = f'{save_num}.{i}.{j}'

            if os.path.exists(save_name(
                    f'{load_num}.{i}.{j}',
                    states_dir,
                    net_name,
            )) and load_num:
                config['training']['network-load'] = f'{load_num}.{i}.{j}'
            else:
                config['training']['network-load'] = 0

            print(
                f"\nSave: {config['training']['network-save']}\n"
                f'Sims: {current_known}\n'
                f'Unknown sims: {current_unknown}\n'
                f"Name: {config['training']['description']}"
            )

            # Remove dataset transform & initialise network
            dataset.images = net.transforms['inputs'](dataset.images, back=True)
            dataset.labels = net.transforms['targets'](dataset.labels, back=True)
            net = net_init(dataset, config=config)
            net.description = config['training']['description']
            net.training(epochs, loaders)
            net.save()
            data = net.predict(loaders[1])
            data['targets'] = dataset.correct_unknowns(data['targets'].squeeze())
            # data['latent'] = net.transforms['targets'](data['preds']).numpy() # For Encoders
            meds, means, stes = analysis.summary(data)[:3]

            results[i]['log_meds'].append(meds)
            results[i]['log_means'].append(means)
            results[i]['log_stes'].append(stes)

            meds = net.transforms['targets'](meds, back=True)
            means, stes = net.transforms['targets'](means, back=True, uncertainty=stes)

            results[i]['meds'].append(meds)
            results[i]['means'].append(means)
            results[i]['stes'].append(stes)
            results[i]['targets'].append(np.unique(data['targets']))
            results[i]['nets'].append(net)

        for key in ('meds', 'means', 'stes', 'log_meds', 'log_means', 'log_stes', 'targets'):
            results[i][key] = np.stack(results[i][key])

    with open(os.path.join(
            config['output']['batch-train-dir'],
            f'batch_train_{save_num}.pkl',
    ), 'wb') as file:
        pickle.dump(results, file)


if __name__ == '__main__':
    main()
