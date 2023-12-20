"""
Main script for training the network
"""
from time import time

import torch
from torch.utils.data import DataLoader

from src.utils.utils import get_device
from src.utils.networks import Encoder, NormFlow


def train_val(loader: DataLoader, network: Encoder | NormFlow) -> float:
    """
    Trains the network for one epoch

    Parameters
    ----------
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    network : Encoder | NormFlow
        Network to use for training/validation

    Returns
    -------
    float
        Average loss value
    """
    epoch_loss = 0
    device = get_device()[1]

    with torch.set_grad_enabled(network.train_state):
        for _, labels, images in loader:
            labels = labels.to(device)
            images = images.to(device)

            epoch_loss += network.loss(images, labels)

    return epoch_loss / len(loader)


def training(
        epochs: tuple[int, int],
        loaders: tuple[DataLoader, DataLoader],
        network: Encoder | NormFlow):
    """
    Trains & validates the network for each epoch

    Parameters
    ----------
    epochs : tuple[integer, integer]
        Initial epoch & number of epochs to train
    loaders : tuple[DataLoader, DataLoader]
        Train and validation dataloaders
    network : Encoder | NormFlow
        Network to use for training
    """
    # Train for each epoch
    for _ in range(*epochs):
        t_initial = time()
        epoch = network.epoch()

        # Train network
        network.train(True)
        network.losses[0].append(train_val(loaders[0], network))

        # Validate network
        network.train(False)
        network.losses[1].append(train_val(loaders[1], network))
        network.scheduler()

        # Save training progress
        network.save()

        print(f'Epoch [{epoch}/{epochs[1]}]\t'
              f'Training loss: {network.losses[0][-1]:.3e}\t'
              f'Validation loss: {network.losses[1][-1]:.3e}\t'
              f'Time: {time() - t_initial:.1f}')

    network.train(False)
    final_loss = train_val(loaders[1], network)
    print(f'\nFinal validation loss: {final_loss:.3e}')
