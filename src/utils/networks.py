"""
Classes for training the networks
"""
import os
import logging as log

import torch
from zuko.flows import NSF
from torch import nn, Tensor
from netloader.network import Network

from src.utils.utils import get_device, save_name


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
            state = torch.load(path, map_location=get_device()[1])

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


class Autoencoder(BaseNetwork):
    """
    Network handler for autoencoder type networks

    Attributes
    ----------
    network : Network
        Autoencoder network
    train_state : boolean, default = True
        If network should be in the train or eval state
    latent_loss : float, default = 1e-2
        Loss weight for the latent MSE loss
    bound_loss : float, default = 1e-3
        Loss weight for the latent bounds loss
    description : string, default = ''
        Description of the network training
    transform : tuple[float, float], default = None
        Expected label transformation of the autoencoder
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current autoencoder training and validation losses

    Methods
    -------
    loss(high_dim, low_dim) -> Tensor
        Calculates the loss from the autoencoder's predictions
    """
    def __init__(
            self,
            save_num: int,
            states_dir: str,
            network: Network,
            description: str = ''):
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
        super().__init__(save_num, states_dir, network, description=description)
        self.latent_loss = 1e-2
        self.bound_loss = 1e-3

    def loss(self, high_dim: Tensor, low_dim: Tensor) -> Tensor:
        """
        Calculates the loss from the autoencoder's predictions

        Parameters
        ----------
        high_dim : Tensor
            High dimensional data as input
        low_dim : Tensor
            Low dimensional data as the latent target

        Returns
        -------
        Tensor
            Loss from the autoencoder's predictions'
        """
        bounds = torch.tensor([0., 1.]).to(get_device()[1])
        output = self.network(high_dim)
        latent = self.network.clone

        loss = nn.MSELoss()(output, high_dim) + self.network.kl_loss

        if self.latent_loss:
            loss += self.latent_loss * nn.MSELoss()(latent, low_dim)

        if self.bound_loss:
            loss += self.bound_loss * torch.mean(torch.cat((
                (bounds[0] - latent) ** 2 * (latent < bounds[0]),
                (latent - bounds[1]) ** 2 * (latent > bounds[1]),
            )))

        if self.train_state:
            self.network.optimiser.zero_grad()
            loss.backward()
            self.network.optimiser.step()

        return loss.item()


class Decoder(BaseNetwork):
    """
    Calculates the loss for a network that takes low-dimensional data and predicts
    high-dimensional data

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
    loss(high_dim, low_dim) -> Tensor
        Calculates the loss from the network's predictions'
    """
    def loss(self, high_dim: Tensor, low_dim: Tensor) -> Tensor:
        """
        Calculates the loss from the network's predictions'

        Parameters
        ----------
        high_dim : Tensor
            High dimensional data as the target
        low_dim : Tensor
            Low dimensional data as input

        Returns
        -------
        Tensor
            Loss from the network's predictions'
        """
        output = self.network(low_dim)
        loss = nn.MSELoss()(output, high_dim)

        # Update network
        if self.train_state:
            self.network.optimiser.zero_grad()
            loss.backward()
            self.network.optimiser.step()

        return loss.item()


class Encoder(BaseNetwork):
    """
    Calculates the loss for a network that takes high-dimensional data
    and predicts low-dimensional data

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
    loss(high_dim, low_dim) -> Tensor
        Calculates the loss from the network's predictions'
    """
    def __init__(
            self,
            save_num: int,
            states_dir: str,
            network: Network,
            description: str = '',
            loss_function: nn.Module = nn.MSELoss()):
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
        loss_function : Module, default = MSELoss
            Loss function to use
        """
        super().__init__(save_num, states_dir, network, description=description)
        self._loss_function = loss_function

    def loss(self, high_dim: Tensor, low_dim: Tensor) -> Tensor:
        """
        Calculates the loss from the network's predictions'

        Parameters
        ----------
        high_dim : Tensor
            High dimensional data as input
        low_dim : Tensor
            Low dimensional data as the target

        Returns
        -------
        Tensor
            Loss from the network's predictions'
        """
        output = self.network(high_dim)

        # Default shape is (N, L), but cross entropy expects (N)
        if isinstance(self._loss_function, nn.CrossEntropyLoss):
            low_dim = low_dim.squeeze()

        loss = self._loss_function(output, low_dim)

        # Update network
        if self.train_state:
            self.network.optimiser.zero_grad()
            loss.backward()
            self.network.optimiser.step()

        return loss.item()

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
        output = super().predict(high_dim)

        if isinstance(self._loss_function, nn.CrossEntropyLoss):
            output = torch.max(output, dim=-1, keepdim=True)[1]

        return output


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
    loss(high_dim, low_dim) -> Tensor
        Calculates the loss from the network and flow's predictions'
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

    def loss(self, high_dim: Tensor, low_dim: Tensor) -> Tensor:
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
        Tensor
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
