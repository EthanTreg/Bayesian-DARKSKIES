"""
Searches the parameter space to find the performance of the network
"""
import os
import pickle
from time import time
from typing import Any, BinaryIO

import torch
import numpy as np
from torch import optim, nn
from torch.nn import Module
from torch.utils.data import DataLoader
from netloader.utils.utils import get_device, progress_bar
from netloader.network import Network
from netloader import layers

from src.main import init
from src.optimise import _update_net
from src.utils.utils import open_config
from src.utils.clustering import CompactClusterEncoder


def _objective(
        idx: int,
        latent_dim: int,
        loaders: tuple[DataLoader, DataLoader],
        config_path: str = '../config.yaml') -> tuple[float, float, CompactClusterEncoder]:
    """
    Optuna objective to pick hyperparameters and train the network

    Parameters
    ----------
    idx : int
        Layer index for the latent space
    latent_dim : int
        Dimensions of the latent space
    loaders : tuple[DataLoader, DataLoader]
        Training and validation dataloaders
    config_path : str, default = '../config.yaml'
        Path to the configuration file

    Returns
    -------
    tuple[float, float, CompactClusterEncoder]
        Performance metrics of the trial and the network
    """
    epochs: int
    smooth: int
    initial_time: float
    learning_rate: float
    name: str
    nets_dir: str
    losses: list[float] = []
    data: dict[str, np.ndarray]
    config: dict[str, Any]
    main_config: dict[str, Any]
    device: torch.device = get_device()[1]
    net: CompactClusterEncoder

    _, main_config = open_config('main', config_path)
    _, config = open_config('optimise', config_path)
    nets_dir = main_config['data']['network-configs-directory']
    epochs = config['optimisation']['epochs']
    smooth = config['optimisation']['smooth-number']
    learning_rate = config['optimisation']['learning-rate']
    name = config['optimisation']['network-name']
    net = torch.load(config['output']['base-network'])

    # Load network configuration file
    _update_net(idx, latent_dim, name, nets_dir)

    # Create network
    net.net = Network(
        'optuna_config',
        nets_dir,
        list(loaders[1].dataset.dataset[0][2].shape),
        [len(torch.unique(loaders[1].dataset.dataset.labels))],
    )
    net._epoch = 0
    net.optimiser = optim.AdamW(net.net.parameters(), lr=learning_rate)
    net.scheduler = optim.lr_scheduler.ReduceLROnPlateau(net.optimiser, factor=0.5)
    net.to(device)

    # Train network
    for i in range(net._epoch, epochs):
        initial_time = time()

        # Training
        net.train(True)
        net._train_val(loaders[0])

        # Validation
        data = net.predict(loaders[1])
        net.losses[1].append(np.count_nonzero(
            data['targets'].squeeze() == data['preds']
        ) / len(data['targets']))

        # Update
        net._update_scheduler(metrics=net.losses[1][-1])
        net._update_epoch()
        losses.append(float(np.mean(net.losses[1][-smooth:])))
        progress_bar(
            i,
            epochs,
            text=f'Epoch {i + 1}/{epochs}\tAccuracy: {net.losses[1][-1]:.1%}\tTraining time: '
                 f'{time() - initial_time:.1f} s',
            flush=True,
        )

        # End plateaued networks early
        if (net._epoch > net.scheduler.patience * 2 and
                losses[-net.scheduler.patience * 2] > losses[-1]):
            print('Trial plateaued, ending early...')
            break

    return losses[-1], torch.nn.MSELoss()(
        net.header['targets'](data['targets']).squeeze(),
        torch.from_numpy(data['latent'][:, 0]),
    ).item(), net


def main(config_path: str = '../config.yaml') -> None:
    """
    Main function for hyperparameter search

    Parameters
    ----------
    config_path : str, default = '../config.yaml'
        Path to the configuration file
    """
    i: int
    idx: int
    dim: int
    study_save: int
    repeats: int = 5
    run_idx: int = 0
    study_dir: str
    sim_dims: list[int]
    sim: list[str]
    current_sims: list[str] = []
    latent_dims: list[list[int]] = [
        *[[4, 5, 6, 8, 9, 12, 14, 16, 18]] * 2,
        *[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 50, 100]] * 2,
    ]
    sims: list[list[str]] = [
        ['bahamas_cdm', 'bahamas_0.1', 'bahamas_0.3', 'bahamas_1'],
        ['bahamas_cdm_low', 'bahamas_cdm_hi'],
        ['flamingo'],
        ['flamingo_low', 'flamingo_hi'],
        # ['tng'],
        # ['darkskies_0.01', 'darkskies_0.05', 'darkskies_0.1', 'darkskies_0.2'],
    ]
    losses: tuple[float, float]
    loaders: tuple[DataLoader, DataLoader]
    data: dict[int, dict[str, Any]] = {}
    config: dict[str, Any]
    main_config: dict[str, Any]
    file: BinaryIO
    module: Module
    net: CompactClusterEncoder

    _, config = open_config('optimise', config_path)
    _, main_config = open_config('main', config_path)
    main_config['training'] |= config['optimisation']

    study_save = config['optimisation']['study-save']
    study_load = config['optimisation']['study-load']
    study_dir = config['data']['study-directory']

    if os.path.exists(f'{study_dir}study_{study_load}.pkl'):
        with open(f'{study_dir}study_{study_load}.pkl', 'rb') as file:
            data = pickle.load(file)

    # Loop through sims
    for i, (sim, sim_dims) in enumerate(zip(sims, latent_dims)):
        print(f'Sim Test {i + 1}/{len(sims)}', flush=True)
        current_sims += sim
        loaders, net, _ = init(config=main_config, known=current_sims)
        net.save_path = config['output']['network']
        net._verbose = None
        torch.save(net, config['output']['base-network'])

        # Find the last checkpoint corresponding to the latent space
        for idx, module in enumerate(net.net.net[::-1]):
            if isinstance(module, layers.Checkpoint):
                idx = len(net.net.net) - 2 - idx
                break

        # Loop through latent dimensions
        for dim in sim_dims:
            if (
                    run_idx in data and
                    all(sim in data[run_idx]['sims'] for sim in current_sims) and
                    dim <= data[run_idx]['latent_dim'] and
                    len(data[run_idx]['losses']) >= repeats):
                continue

            if len(data[run_idx]['losses']) >= repeats:
                run_idx += 1
                data[run_idx] = {'latent_dim': dim, 'sims': current_sims, 'losses': [], 'nets': []}

            # Repeat n times
            for _ in range(repeats - len(data[run_idx]['losses'])):
                *losses, net = _objective(idx, dim, loaders)
                data[run_idx]['losses'].append(losses)
                data[run_idx]['nets'].append(net)

                if study_save:
                    with open(f'{study_dir}study_{study_save}.pkl', 'wb') as file:
                        pickle.dump(data, file)


if __name__ == "__main__":
    main()
