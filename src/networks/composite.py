"""
Classes that contain multiple types of networks
"""
import logging as log
import os

import torch
from netloader.network import Network
from torch import Tensor
from zuko.flows import NSF

from src.utils.utils import save_name
from src.networks.base import BaseNetwork


class NormFlow(BaseNetwork):
    """
    Calculates the loss for a network and normalising flow that takes high-dimensional data
    and predicts a low-dimensional data distribution

    Attributes
    ----------
    flow : NSF
        Normalising flow to predict low-dimensional data distribution
    network : Network
        Network to condition the normalising flow from high-dimensional data
    train_state : boolean, default = True
        If network should be in the train or eval state
    train_flow : boolean, default = True
        If normalising flow should be trained
    train_net : boolean, default = False
        If network should be trained
    description : string, default = ''
        Description of the network training
    transform : tuple[float, float], default = None
        Expected label transformation of the flow
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current flow training and validation losses

    Methods
    -------
    train()
        Flips the train/eval state of the network and flow
    load(states_dir, load_num) -> ndarray | None
        Loads the flow and network from a previously saved state, if load_num != 0
    epoch() -> integer
        Updates network and flow epoch if they are being trained
    save(indices)
        If save_num is provided, saves the network to the states directory
    scheduler()
        Updates the scheduler for the flow and/or network if they are being trained
    predict(data)
        Generates probability distributions for the given data
    """
    def __init__(
            self,
            states_dir: str,
            save_num: tuple[int, int],
            flow: NSF,
            network: Network,
            net_layers: int = None,
            description: str = ''):
        """
        Parameters
        ----------
        states_dir : string
            Directory to save the network and flow
        save_num : tuple[integer]
            File numbers to save the flow and network
        flow : NSF
            Normalising flow to predict low-dimensional data distribution
        network : Network
            Network to condition the normalising flow from high-dimensional data
        net_layers : integer, default = None
            Number of layers in the network to use, if None use all layers
        description : string, default = ''
            Description of the network training
        """
        super().__init__(save_num[0], states_dir, network, description=description)
        self._net_layers = net_layers
        self.train_net = False
        self.train_flow = True
        self.flow = flow
        self.network = network
        self.flow.epoch = 0

        if save_num[1]:
            self.flow.save_path = save_name(save_num[1], states_dir, self.flow.name)

            if os.path.exists(self.flow.save_path):
                log.warning(f'{self.flow.save_path} already exists and will be overwritten '
                            f'if training continues')
        else:
            self.flow.save_path = None

    def _loss(self, high_dim: Tensor, low_dim: Tensor) -> float:
        """
        Calculates the loss from the network and flow's predictions'

        Parameters
        ----------
        high_dim : Tensor
            High dimensional data as input
        low_dim : Tensor
            Low dimensional data as the target

        Returns
        -------
        float
            Loss from the flow's predictions'
        """
        # Temporarily truncate the network
        self.network.layer_num = self._net_layers

        # Normalising flow loss
        output = self.network(high_dim)
        loss = -self.flow(output).log_prob(low_dim).mean()

        if not self.train_state:
            self.network.layer_num = None
            return loss.item()

        # Update network
        self.flow.optimiser.zero_grad()
        self.network.optimiser.zero_grad()
        loss.backward()

        if self.train_net:
            self.network.optimiser.step()

        if self.train_flow:
            self.flow.optimiser.step()

        # Remove network truncation
        self.network.layer_num = None

        return loss.item()

    def train(self, train: bool):
        """
        Sets the train/eval state of the network/flow

        Parameters
        ----------
        train : boolean
            If the network/flow should be in the train state
        """
        super().train(train)

        if self.train_state:
            self.flow.train()
        else:
            self.flow.eval()

    def load(self, states_dir: str, load_num: tuple[int, int]):
        """
        Loads the flow and network from a previously saved state, if load_num != 0

        Can account for changes in the network/flow

        Parameters
        ----------
        states_dir : string
            Directory to the save files
        load_num : tuple[integer, integer]
            File numbers of the network and flow saved state
        """
        super().load(load_num[0], states_dir)
        super().load(load_num[1], states_dir, network=self.flow)

    def epoch(self) -> int:
        """
        Updates network and flow epoch if they are being trained

        Returns
        -------
        integer
            Epoch number
        """
        if self.train_flow:
            self.flow.epoch += 1

        if self.train_net:
            self.network.epoch += 1

        if self.train_flow:
            return self.flow.epoch

        return self.network.epoch

    def save(self):
        """
        If save_num is provided, saves the network to the states directory
        """
        if self.train_net:
            super().save(self.network)

        if self.train_flow:
            self.flow.losses = self.losses
            super().save(self.flow)

    def scheduler(self):
        """
        Updates the scheduler for the flow and/or network if they are being trained
        """
        if self.train_net:
            super().scheduler()

        if self.train_flow:
            self.flow.scheduler.step(self.losses[1][-1])

    def predict(self, high_dim: Tensor) -> Tensor:
        """
        Generates probability distributions for the data

        Parameters
        ----------
        high_dim : Tensor
            Data to generate distributions for

        Returns
        -------
        Tensor
            Probability distributions for the given data
        """
        # Temporarily truncate the network and generate samples
        self.network.layer_num = self._net_layers
        samples = torch.transpose(
            self.flow(self.network(high_dim)).sample((1000,)).squeeze(-1),
            0,
            1,
        )

        # Remove network truncation
        self.network.layer_num = None
        return samples
