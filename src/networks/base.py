"""
Base network class to base other networks off
"""
import os
import logging as log
from time import time

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from netloader.network import Network
from zuko.flows import NSF
from numpy import ndarray

from src.utils.utils import save_name, get_device


class BaseNetwork:
    """
    Base network class that other types of networks build from

    Attributes
    ----------
    network : Network
        Neural network
    train_state : boolean, default = True
        If network should be in the train or eval state
    description : string, default = ''
        Description of the network training
    transform : tuple[float, float], default = None
        Expected low-dimensional data transformation of the network
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current network training and validation losses

    Methods
    -------
    train(train)
        Flips the train/eval state of the network
    load(load_num, states_dir, network=None) -> ndarray | None
        Loads the network from a previously saved state
    epoch() -> integer
        Updates network epoch
    training(epoch, training)
        Trains & validates the network for each epoch
    save()
        If save_num is provided, saves the network to the states directory
    scheduler()
        Updates the scheduler for the network
    predict(loader, save, path) -> tuple[ndarray, ndarray, ndarray]
        Generates predictions for a dataset and can save to a file
    batch_predict(high_dim) -> Tensor
        Generates predictions for the given data batch
    """
    def __init__(self, save_num: int, states_dir: str, network: Network, description: str = ''):
        """
        Parameters
        ----------
        save_num : integer
            File number to save the network
        states_dir : string
            Directory to save the network
        network : Network
            Network to predict low-dimensional data
        description : string, default = ''
            Description of the network training
        """
        self._device = get_device()[1]
        self.train_state = True
        self.description = description
        self.transform = None
        self.losses = ([], [])
        self.idxs = None
        self.network = network
        self.network.epoch = 0

        if save_num:
            self.save_path = save_name(save_num, states_dir, self.network.name)

            if os.path.exists(self.save_path):
                log.warning(f'{self.save_path} already exists and will be overwritten '
                            f'if training continues')
        else:
            self.save_path = None

    def _train_val(self, loader: DataLoader) -> float:
        """
        Trains the network for one epoch

        Parameters
        ----------
        loader : DataLoader
            PyTorch DataLoader that contains data to train

        Returns
        -------
        float
            Average loss value
        """
        epoch_loss = 0

        with torch.set_grad_enabled(self.train_state):
            for _, labels, images, *_ in loader:
                labels = labels.to(self._device)
                images = images.to(self._device)

                epoch_loss += self._loss(images, labels)

        return epoch_loss / len(loader)

    def _loss(self, high_dim: Tensor, low_dim: Tensor) -> float:
        """
        Empty method for child classes to base their loss functions on

        Parameters
        ----------
        high_dim : Tensor
            High dimension data
        low_dim : Tensor
            Low dimension data

        Returns
        -------
        Tensor
            Loss
        """

    def _update(self, loss: Tensor):
        """
        Updates the network using backpropagation

        Parameters
        ----------
        loss : Tensor
            Loss to perform backpropagation from
        """
        if self.train_state:
            self.network.optimiser.zero_grad()
            loss.backward()
            self.network.optimiser.step()

    def train(self, train: bool):
        """
        Flips the train/eval state of the network

        Parameters
        ----------
        train : boolean
            If the network should be in the train state
        """
        self.train_state = train

        if self.train_state:
            self.network.train()
        else:
            self.network.eval()

    def load(self, load_num: int, states_dir: str, network: NSF | Network = None):
        """
        Loads the network from a previously saved state, if load_num != 0

        Can account for changes in the network

        Parameters
        ----------
        load_num : integer
            File number of the saved state
        states_dir : string
            Directory to the save files
        network : NSF | Network, default = self.network
            Network to load

        Returns
        -------
        ndarray | None
            Dataset indices
        """
        if not load_num:
            return

        if network is None:
            network = self.network

        path = save_name(load_num, states_dir, network.name)

        try:
            state = torch.load(path, map_location=self._device)

            # Apply the saved states to the new network
            if 'network' in state:
                self.idxs = state['indices']
                self.transform = state['transform']
                del state['network'].save_path
                network.__dict__.update(state['network'].__dict__)
                self.losses = network.losses
            else:
                network.epoch = state['epoch']
                network.losses = (state['train_loss'], state['val_loss'])
                network.idxs = state['indices']
                network.load_state_dict(network.state_dict() | state['state_dict'])
                network.optimiser.load_state_dict(state['optimiser'])
                network.scheduler.load_state_dict(state['scheduler'])
        except FileNotFoundError:
            log.warning(f'{path} dose not exist\nNo state will be loaded')

    def epoch(self) -> int:
        """
        Updates network epoch

        Returns
        -------
        integer
            Epoch number
        """
        self.network.epoch += 1
        return self.network.epoch

    def training(self, epoch: int, loaders: tuple[DataLoader, DataLoader]):
        """
        Trains & validates the network for each epoch

        Parameters
        ----------
        epoch : integer
            Number of epochs to train the network up to
        loaders : tuple[DataLoader, DataLoader]
            Train and validation dataloaders
        """
        # Train for each epoch
        for _ in range(self.network.epoch, epoch):
            t_initial = time()

            # Train network
            self.train(True)
            self.losses[0].append(self._train_val(loaders[0]))

            # Validate network
            self.train(False)
            self.losses[1].append(self._train_val(loaders[1]))
            self.scheduler()

            # Save training progress
            self.epoch()
            self.save()

            print(f'Epoch [{self.network.epoch}/{epoch}]\t'
                  f'Training loss: {self.losses[0][-1]:.3e}\t'
                  f'Validation loss: {self.losses[1][-1]:.3e}\t'
                  f'Time: {time() - t_initial:.1f}')

        self.train(False)
        final_loss = self._train_val(loaders[1])
        print(f'\nFinal validation loss: {final_loss:.3e}')

    def scheduler(self):
        """
        Updates the scheduler for the network
        """
        self.network.scheduler.step(self.losses[1][-1])

    def save(self):
        """
        Saves the network to the given path
        """
        if self.save_path:
            torch.save(self, self.save_path)

    def predict(
            self,
            loader: DataLoader,
            path: str = None,
            **kwargs) -> tuple[ndarray, ndarray, ndarray]:
        """
        Generates predictions for a dataset and can save to a file

        Parameters
        ----------
        loader : DataLoader
            Dataset to generate predictions for
        path : string, default = None
            Path to save the predictions if they should be saved
        **kwargs
            Optional keyword arguments to pass into batch_predict

        Returns
        -------
        tuple[N ndarray, Nx... ndarray, Nx... ndarray]
            Prediction IDs, target values and predicted values for dataset of size N
        """
        initial_time = time()
        header = 'IDs,Targets,Predictions'
        ids = []
        targets = []
        predictions = []
        self.train(False)

        # Generate predictions
        with torch.no_grad():
            for id_batch, target, images, *_ in loader:
                ids.extend(id_batch)
                targets.extend(target)
                predictions.extend(self.batch_predict(images.to(self._device), **kwargs).cpu())

        # Transform values
        ids = np.array(ids)
        targets = torch.stack(targets).numpy() * self.transform[1] + self.transform[0]
        predictions = torch.stack(predictions).numpy() * self.transform[1] + self.transform[0]

        if path:
            output = np.hstack((ids[:, np.newaxis], targets, predictions))
            print(f'Prediction time: {time() - initial_time:.3e} s')
            np.savetxt(path, output, delimiter=',', fmt='%s', header=header)

        return ids, targets, predictions

    def batch_predict(self, data: Tensor) -> Tensor:
        """
        Generates predictions for the given data batch

        Parameters
        ----------
        data : Nx... Tensor
            N data to generate predictions for

        Returns
        -------
        Nx... Tensor
            N predictions for the given data
        """
        return self.network(data)


def load_net(num: int, states_dir: str, net_name: str) -> BaseNetwork:
    """
    Loads a network from file

    Parameters
    ----------
    num : integer
        File number of the saved state
    states_dir : string
        Directory to the save files
    net_name : string
        Name of the network

    Returns
    -------
    BaseNetwork
        Saved network object
    """
    path = save_name(num, states_dir, net_name)
    return torch.load(path, map_location=get_device()[1])
