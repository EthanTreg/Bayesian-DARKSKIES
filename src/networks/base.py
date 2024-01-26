"""
Base network class to base other networks off
"""
import logging as log
import os
from time import time

import torch
from netloader.network import Network
from torch import Tensor
from torch.utils.data import DataLoader
from zuko.flows import NSF

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
    save(network=None)
        If save_num is provided, saves the network to the states directory
    scheduler()
        Updates the scheduler for the network
    predict(high_dim)
        Generates predictions for the given data
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
            self.network.save_path = save_name(save_num, states_dir, self.network.name)

            if os.path.exists(self.network.save_path):
                log.warning(f'{self.network.save_path} already exists and will be overwritten '
                            f'if training continues')
        else:
            self.network.save_path = None

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

    def save(self, network: NSF | Network = None):
        """
        Saves the network to the given path

        Parameters
        ----------
        network : Network | NSF, default = self.network
            Network to save
        """
        if network is None:
            network = self.network
            network.losses = self.losses

        if network.save_path is None:
            return

        torch.save({
            'description': self.description,
            'transform': self.transform,
            'indices': self.idxs,
            'network': network,
        }, network.save_path)

    def predict(self, high_dim: Tensor) -> Tensor:
        """
        Generates predictions for the given data

        Parameters
        ----------
        high_dim : Tensor
            Data to generate predictions for

        Returns
        -------
        Tensor
            Predictions for the given data
        """
        return self.network(high_dim)
