"""
Main script for training the network
"""
from time import time

import torch
import normflows as nf
from numpy import ndarray
from torch import nn, Tensor
from torch.utils.data import DataLoader
from netloader.network import Network

from src.utils.utils import get_device


def train_val(
        loader: DataLoader,
        network: Network,
        train: bool = True) -> tuple[float, ndarray, ndarray]:
    """
    Trains the encoder or decoder using cross entropy or mean squared error

    Parameters
    ----------
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    network : Network
        Network to use for training/validation
    train : bool, default = True
        If network should be trained or validated

    Returns
    -------
    tuple[float, ndarray, ndarray]
        Average loss value, labels, & network output
    """
    epoch_loss = 0
    device = get_device()[1]

    if train:
        network.train()
    else:
        network.eval()

    with torch.set_grad_enabled(train):
        for _, labels, images in loader:
            labels = labels.to(device)
            images = images.to(device)

            output = network(images)
            loss = nn.MSELoss()(output, labels)

            if train:
                # Optimise network
                network.optimiser.zero_grad()
                loss.backward()
                network.optimiser.step()

            epoch_loss += loss.item()

    return epoch_loss / len(loader), labels.cpu().numpy(), output.detach().cpu().numpy()


def training(
        epochs: tuple[int, int],
        loaders: tuple[DataLoader, DataLoader],
        network: Network,
        save_num: int = 0,
        states_dir: str = None,
        losses: tuple[list[Tensor], list[Tensor]] = None
        ) -> tuple[tuple[list, list], ndarray, ndarray]:
    """
    Trains & validates the network for each epoch

    Parameters
    ----------
    epochs : tuple[integer, integer]
        Initial epoch & number of epochs to train
    loaders : tuple[DataLoader, DataLoader]
        Train and validation dataloaders
    network : Network
        Network to use for training
    save_num : integer, default = 0
        The file number to save the new state, if 0, nothing will be saved
    states_dir : string, default = None
        Path to the folder where the network state will be saved, not needed if save_num = 0
    losses : tuple[list, list], default = ([], [])
        Train and validation losses for each epoch, can be empty

    Returns
    -------
    tuple[tuple[list, list], ndarray, ndarray]
        Train & validation losses, labels, & network outputs
    """
    if not losses:
        losses = ([], [])

    # Train for each epoch
    for epoch in range(*epochs):
        t_initial = time()
        epoch += 1

        # Train network
        losses[0].append(train_val(loaders[0], network)[0])

        # Validate network
        losses[1].append(train_val(loaders[1], network, train=False)[0])
        network.scheduler.step(losses[1][-1])

        # Save training progress
        if save_num:
            state = {
                'epoch': epoch,
                'train_loss': losses[0],
                'val_loss': losses[1],
                'indices': loaders[0].dataset.dataset.indices,
                'state_dict': network.state_dict(),
                'optimiser': network.optimiser.state_dict(),
                'scheduler': network.scheduler.state_dict(),
            }

            torch.save(state, f'{states_dir}{network.name}_{save_num}.pth')

        print(f'Epoch [{epoch}/{epochs[1]}]\t'
              f'Training loss: {losses[0][-1]:.3e}\t'
              f'Validation loss: {losses[1][-1]:.3e}\t'
              f'Time: {time() - t_initial:.1f}')

    # Final validation
    loss, labels, outputs = train_val(loaders[1], network, train=False)
    losses[1].append(loss)
    print(f'\nFinal validation loss: {losses[1][-1]:.3e}')

    return losses, labels, outputs


# def nf_training():
#     base = nf.distributions.base.DiagGaussian()
