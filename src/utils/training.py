"""
Main script for training the network
"""
from time import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.utils.utils import get_device
from src.utils.networks import NeuralNetwork, NormFlow


def train_val(loader: DataLoader, network: NeuralNetwork | NormFlow) -> float:
    """
    Trains the network for one epoch

    Parameters
    ----------
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    network : NeuralNetwork | NormFlow
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
        network: NeuralNetwork | NormFlow,
        losses: tuple[list[Tensor], list[Tensor]] = None
        ) -> tuple[list[Tensor], list[Tensor]]:
    """
    Trains & validates the network for each epoch

    Parameters
    ----------
    epochs : tuple[integer, integer]
        Initial epoch & number of epochs to train
    loaders : tuple[DataLoader, DataLoader]
        Train and validation dataloaders
    network : NeuralNetwork | NormFlow
        Network to use for training
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Train and validation losses for each epoch, can be empty

    Returns
    -------
    tuple[list[Tensor], list[Tensor]]
        Train & validation losses
    """
    if not losses:
        losses = ([], [])

    # Train for each epoch
    for epoch in range(*epochs):
        t_initial = time()
        epoch += 1

        # Train network
        network.train(True)
        losses[0].append(train_val(loaders[0], network))

        # Validate network
        try:
            network.num_correct = 0
        except AttributeError:
            pass

        network.train(False)
        losses[1].append(train_val(loaders[1], network))
        network.scheduler(losses[1][-1])

        try:
            network.accuracy.append(network.num_correct.cpu().item() / len(loaders[1].dataset))
            accuracy = f'Accuracy: {network.accuracy[-1] * 100:.1f}%\t'
        except AttributeError:
            accuracy = ''

        # Save training progress
        network.save(epoch, losses, loaders[0].dataset.dataset.indices)

        print(f'Epoch [{epoch}/{epochs[1]}]\t'
              f'{accuracy}'
              f'Training loss: {losses[0][-1]:.3e}\t'
              f'Validation loss: {losses[1][-1]:.3e}\t'
              f'Time: {time() - t_initial:.1f}')

    # Final validation
    network.train(False)
    losses[1].append(train_val(loaders[1], network))
    print(f'\nFinal validation loss: {losses[1][-1]:.3e}')

    return losses
