"""
Optimises the hyperparameters of networks
"""
import os
import json
from time import time

import torch
import optuna
import joblib
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from netloader.network import Network
from optuna import pruners

from src.main import init
from src.utils.utils import get_device, open_config


def _update_net(
        latent_dim: int,
        latent_num: int,
        class_num: int,
        name: str,
        nets_dir: str,
        idxs: tuple[int, int]):
    """
    Updates the network configuration with latent dimension and optional layers between feature and
    cluster latent spaces, and latent and classification latent spaces

    Parameters
    ----------
    latent_dim : integer
        Dimension of the latent space
    latent_num : integer
        Number of linear layers between the feature and cluster latent space
    class_num : integer
        Number of linear layers between the cluster and classification latent spaces
    name : string
        Name of the network
    nets_dir : string
        Directory of network configurations
    idxs : tuple[integer, integer]
        Indices of the checkpoints for the feature and cluster latent spaces
    """
    with open(f'{nets_dir}{name}.json', 'r', encoding='utf-8') as file:
        net_config = json.load(file)

    net_config['layers'][idxs[0] + 1]['features'] = latent_dim

    for _ in range(class_num):
        net_config['layers'].insert(idxs[1] + 1, {'type': 'linear', 'features': 64})

    for _ in range(latent_num):
        net_config['layers'].insert(idxs[0] + 1, {'type': 'linear', 'features': 256})

    with open('../data/optuna_config.json', 'w', encoding='utf-8') as file:
        json.dump(net_config, file)


def _objective(
        epochs: int,
        idxs: tuple[int, int],
        loaders: tuple[DataLoader, DataLoader],
        trial: optuna.Trial) -> float:
    """
    Optuna objective to pick hyperparameters and train the network

    Parameters
    ----------
    epochs : integer
        Number of epochs to train the network for
    idxs : tuple[integer, integer]
        Indices of the checkpoints for the feature and cluster latent spaces
    loaders : tuple[DataLoader, DataLoader]
        Training and validation dataloaders
    trial : Trial
        Optuna trial

    Returns
    -------
    float
        Performance metric of the trial
    """
    device = get_device()[1]
    net = torch.load('../data/base_net.pth')
    _, config = open_config('main', '../config.yaml')
    name = config['training']['network-name']
    nets_dir = config['data']['network-configs-directory']

    # Latent dimension
    latent_dim = trial.suggest_int('latent_dim', 1, 10)

    # Learning rate
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)

    # Loss weighting
    cluster_loss = trial.suggest_float('cluster_loss', 1e-1, 10, log=True)
    class_loss = trial.suggest_float('class_loss', 1e-1, 10, log=True)

    # Linear layers
    latent_num = trial.suggest_int('latent_layers', 0, 3)
    class_num = trial.suggest_int('class_layers', 0, 3)

    # Load network configuration file
    _update_net(latent_dim, latent_num, class_num, name, nets_dir, idxs)

    # Create network
    net.network = Network(
        list(loaders[1].dataset.dataset[0][2].shape),
        [len(torch.unique(loaders[1].dataset.dataset.labels))],
        learning_rate,
        'optuna_config',
        '../data/',
    ).to(device)
    net.network.epoch = 0
    net.network.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        net.network.optimiser,
        factor=0.5,
        mode='max',
        verbose=True,
    )

    # Update network hyperparameters
    net.cluster_loss = cluster_loss
    net.class_loss = class_loss

    # Train network
    for i in range(net.network.epoch, epochs):
        initial_time = time()

        # Training
        net.train(True)
        net._train_val(loaders[0])
        print(f'Epoch {i + 1}/{epochs}\tTraining time: {time() - initial_time:.1f}\t', end='')

        # Validation
        _, targets, predicts, *_ = net.predict(loaders[1])
        net.losses[1].append(np.count_nonzero(targets == predicts) / len(targets))

        # Update
        net.scheduler()
        net.epoch()
        trial.report(net.losses[1][-1], i)

        # Prune poor performing networks
        if trial.should_prune() or (
            net.network.epoch > 10 and net.losses[1][-1] < 1 / len(net._classes)
        ):
            raise optuna.TrialPruned()

        # End plateaued networks early
        if len(net.losses[1]) > 20 and net.losses[1][-20] > net.losses[1][-1]:
            return np.mean(net.losses[1][-5:])

    return np.mean(net.losses[1][-5:])


def main():
    """
    Main function for network optimisation
    """
    load = True
    epochs = 100
    trials = 30
    study_file = '../data/net_study.pkl'
    check_idxs = []
    loaders, net = init()
    net.save_path = '../data/optuna_net.pth'
    torch.save(net, '../data/base_net.pth')

    # Find network checkpoints
    for i, layer in enumerate(net.network.layers):
        if layer['type'] == 'checkpoint':
            check_idxs.append(i)

    # Load existing study, or create new study
    if load and os.path.exists(load):
        study = joblib.load(study_file)
    else:
        study = optuna.create_study(
            study_name='Study 1',
            direction='maximize',
            pruner=pruners.PatientPruner(pruners.MedianPruner(n_warmup_steps=10), patience=2),
        )

    # Trial different network configurations
    for _ in range(trials):
        study.optimize(
            lambda trial: _objective(epochs, check_idxs[-2:], loaders, trial),
            n_trials=1,
        )
        joblib.dump(study, study_file)


if __name__ == '__main__':
    main()
