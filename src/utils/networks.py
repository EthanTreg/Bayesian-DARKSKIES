"""
Classes for training the networks
"""
import logging as log

import torch
from numpy import ndarray
from zuko.flows import NSF
from torch import nn, Tensor
from netloader.network import Network

from src.utils.utils import get_device, save_name


class NeuralNetwork:
    """
    Calculates the loss for a network that takes high-dimensional data
    and predicts low-dimensional data

    Attributes
    ----------
    save_path : string
        Path to save the network if save_num != 0
    accuracy : list[float]
        List of accuracies per epoch
    network : Network
        Network to predict low-dimensional data
    train_state : boolean, default = True
        If network should be in the train or eval state
    epoch : integer, default = 0
        Current epoch of training
    num_correct : integer, default = 0
        Number of correct predictions if using cross entropy loss
    description : string, default = ''
        Description of the network training
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current network training and validation losses
    loss_function : Module, default = MSELoss
        Loss function to use

    Methods
    -------
    train()
        Flips the train/eval state of the network
    load(load_num, states_dir)
        Loads the network from a previously saved state
    loss(high_dim, low_dim)
        Calculates the loss from the network's predictions'
    save(epoch, losses, indices)
        If save_num is provided, saves the network to the states directory
    scheduler(loss)
        Updates the scheduler for the network
    predict(data)
        Generates predictions for the given data
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
        self.train_state = True
        self.epoch = 0
        self.num_correct = 0
        self.description = description
        self.accuracy = []
        self.losses = ([], [])
        self.network = network
        self.loss_function = loss_function

        if save_num:
            self.save_path = save_name(save_num, states_dir, self.network.name)
        else:
            self.save_path = None

    def train(self, train: bool):
        """
        Flips the train/eval state of the network

        Parameters
        ----------
        train : boolean
            If the network/flow should be in the train state
        """
        self.train_state = train

        if self.train_state:
            self.network.train()
        else:
            self.network.eval()

    def load(self, load_num: int, states_dir: str) -> ndarray | None:
        """
        Loads the network from a previously saved state, if load_num != 0

        Can account for changes in the network

        Parameters
        ----------
        load_num : integer
            File number of the saved state
        states_dir : string
            Directory to the save files

        Returns
        -------
        ndarray | None
            Dataset indices
        """
        if not load_num:
            return None

        indices = None
        path = save_name(load_num, states_dir, self.network.name)

        try:
            self.epoch, self.accuracy, self.losses, indices = _load(path, self.network)
        except FileNotFoundError:
            log.warning(f'{path} dose not exist\nNo state will be loaded')

        return indices

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
        if isinstance(self.loss_function, nn.CrossEntropyLoss):
            low_dim = low_dim.squeeze()

        loss = self.loss_function(output, low_dim)

        # Update network
        if self.train_state:
            self.network.optimiser.zero_grad()
            loss.backward()
            self.network.optimiser.step()

        # Calculate accuracy if using cross entropy
        if isinstance(self.loss_function, nn.CrossEntropyLoss):
            self.num_correct += torch.count_nonzero(
                torch.argmax(output, dim=-1) == low_dim,
            ).detach().cpu().item()

        return loss.item()

    def save(self, epoch: int, losses: tuple[list[Tensor], list[Tensor]], indices: ndarray):
        """
        If save_num is provided, saves the network to the states directory

        Parameters
        ----------
        epoch : integer
            Current training epoch
        losses : tuple[list[Tensor], list[Tensor]]
            Training and validation losses
        indices : ndarray
            Data indices for random training & validation datasets
        """
        if self.save_path:
            _save(
                epoch,
                self.save_path,
                losses,
                indices,
                self.network,
                description=self.description,
                accuracy=self.accuracy,
            )

    def scheduler(self, loss):
        """
        Updates the scheduler for the network

        Parameters
        ----------
        loss : Tensor
            Last validation loss
        """
        self.network.scheduler.step(loss)

    def predict(self, data: Tensor) -> Tensor:
        """
        Generates predictions for the given data

        Parameters
        ----------
        data : Tensor
            Data to generate predictions for

        Returns
        -------
        Tensor
            Predictions for the given data
        """
        output = self.network(data)

        if isinstance(self.loss_function, nn.CrossEntropyLoss):
            output = torch.max(output, dim=-1, keepdim=True)[1]

        return output


class NormFlow:
    """
    Calculates the loss for a network and normalising flow that takes high-dimensional data
    and predicts a low-dimensional data distribution

    Attributes
    ----------
    save_path_flow : string
        Path to save the flow if save_num[0] != 0
    save_path_net : string
        Path to save the network if save_num[1] != 0
    flow : NSF
        Normalising flow to predict low-dimensional data distribution
    network : Network
        Network to condition the normalising flow from high-dimensional data
    train_state : boolean, default = True
        If network and flow should be in train or eval state
    train_flow : boolean, default = True
        If normalising flow should be trained
    train_net : boolean, default = False
        If network should be trained
    flow_epoch : integer, default = 0
        Current epoch of flow training
    net_epoch : integer, default = 0
        Current epoch of network training
    net_layers : integer, default = None
        Number of layers in the network to use, if None use all layers
    description : string, default = ''
        Description of the network training
    flow_losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current flow training and validation losses
    net_losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current network training and validation losses

    Methods
    -------
    train()
        Flips the train/eval state of the network and flow
    load(states_dir, load_num)
        Loads the flow and network from a previously saved state, if load_num != 0
    loss(high_dim, low_dim)
        Calculates the loss from the network and flow's predictions'
    save(epoch, losses, indices)
        If save_num is provided, saves the network to the states directory
    scheduler(loss)
        Updates the scheduler for the flow and/or network if they are being trained
    predict(data)
        Generates probability distributions for the given data
    """
    def __init__(
            self,
            states_dir: str,
            save_num: tuple[int, int],
            classes: Tensor,
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
        classes : Tensor
            Classes that the network tries to predict
        flow : NSF
            Normalising flow to predict low-dimensional data distribution
        network : Network
            Network to condition the normalising flow from high-dimensional data
        net_layers : integer, default = None
            Number of layers in the network to use, if None use all layers
        description : string, default = ''
            Description of the network training
        """
        self.train_state = True
        self.train_net = False
        self.train_flow = True
        self.flow_epoch = self.net_epoch = 0
        self.net_layers = net_layers
        self.description = description
        self.flow_losses = ([], [])
        self.net_losses = ([], [])
        self.classes = classes
        self.flow = flow
        self.network = network

        if save_num[0]:
            self.save_path_net = save_name(save_num[0], states_dir, self.network.name)
        else:
            self.save_path_net = None

        if save_num[1]:
            self.save_path_flow = save_name(save_num[1], states_dir, self.flow.name)
        else:
            self.save_path_flow = None

    def train(self, train: bool):
        """
        Sets the train/eval state of the network/flow

        Parameters
        ----------
        train : boolean
            If the network/flow should be in the train state
        """
        self.train_state = train

        if self.train_state:
            self.flow.train()
            self.network.train()
        else:
            self.flow.eval()
            self.network.eval()

    def load(self, states_dir: str, load_num: tuple[int, int]) -> ndarray | None:
        """
        Loads the flow and network from a previously saved state, if load_num != 0

        Can account for changes in the network/flow

        Parameters
        ----------
        states_dir : string
            Directory to the save files
        load_num : tuple[integer, integer]
            File numbers of the network and flow saved state

        Returns
        -------
        ndarray
            Dataset indices
        """
        indices = None

        # Network load
        if load_num[0]:
            net_path = save_name(load_num[0], states_dir, self.network.name)

            try:
                self.net_epoch, _, self.net_losses, indices = _load(net_path, self.network)
            except FileNotFoundError:
                log.warning(f'{net_path} dose not exist\nNo state will be loaded')

        # Flow load
        if load_num[1]:
            flow_path = save_name(load_num[1], states_dir, self.flow.name)

            try:
                self.flow_epoch, _, self.flow_losses, indices = _load(flow_path, self.flow)
            except FileNotFoundError:
                log.warning(f'{flow_path} dose not exist\nNo state will be loaded')

        return indices

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
        self.network.layer_num = self.net_layers

        # Calculate the loss
        output = self.network(high_dim)
        loss = -self.flow(output).log_prob(low_dim).mean()

        if not self.train_state:
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

    def save(self, epoch: int, losses: tuple[list[Tensor], list[Tensor]], indices: ndarray):
        """
        If save_num is provided, saves the network to the states directory

        Parameters
        ----------
        epoch : integer
            Current training epoch
        losses : tuple[list[Tensor], list[Tensor]]
            Training and validation losses
        indices : ndarray
            Data indices for random training & validation datasets
        """
        if self.save_path_flow and self.train_flow:
            _save(
                epoch,
                self.save_path_flow,
                losses,
                indices,
                self.flow,
                description=self.description,
            )

        if self.save_path_net and self.train_net:
            _save(
                epoch + self.net_epoch,
                self.save_path_net,
                losses,
                indices,
                self.network,
                description=self.description,
            )

    def scheduler(self, loss: Tensor):
        """
        Updates the scheduler for the flow and/or network if they are being trained

        Parameters
        ----------
        loss : Tensor
            Last validation loss
        """
        if self.train_flow:
            self.flow.scheduler.step(loss)

        if self.train_net:
            self.network.scheduler.step(loss)

    def predict(self, data: Tensor) -> Tensor:
        """
        Generates probability distributions for the given data

        Parameters
        ----------
        data : Tensor
            Data to generate distributions for

        Returns
        -------
        Tensor
            Probability distributions for the given data
        """
        return torch.transpose(
            self.flow(self.network(data)).sample((1000,)).squeeze(),
            0,
            1,
        )


def _load(
        path: str,
        network: Network | NSF,
) -> tuple[
    int,
    list[float],
    tuple[list[Tensor], list[Tensor]],
    ndarray,
]:
    """
    Loads the network from a previously saved state

    Can account for changes in the network

    Parameters
    ----------
    path : string
        Path to the network save
    network : Network | NSF
        Network to load

    Returns
    -------
    tuple[integer, list[float], tuple[list[Tensor], list[Tensor]], ndarray]
        The initial epoch; accuracy; the training and validation losses; and the dataset indices
    """
    state = torch.load(path, map_location=get_device()[1])

    # Apply the saved states to the new network
    initial_epoch = state['epoch']
    network.load_state_dict(network.state_dict() | state['state_dict'])
    network.optimiser.load_state_dict(state['optimiser'])
    network.scheduler.load_state_dict(state['scheduler'])
    train_loss = state['train_loss']
    val_loss = state['val_loss']
    indices = state['indices']

    try:
        accuracy = state['accuracy']
    except KeyError:
        accuracy = []

    return initial_epoch, accuracy, (train_loss, val_loss), indices


def _save(
        epoch: int,
        save_path: str,
        losses: tuple[list[Tensor], list[Tensor]],
        indices: ndarray,
        network: Network | NSF,
        description: str = '',
        accuracy: list[float] = None):
    """
    Saves the network to the given path

    Parameters
    ----------
    epoch : integer
        Current training epoch
    save_path : string
        Path to save the network
    losses : tuple[list[Tensor], list[Tensor]]
        Training and validation losses
    indices : ndarray
        Data indices for random training & validation datasets
    network : Network | NSF
        Network to save
    description : string, default = ''
        Description of the network training
    accuracy : list[float], default = None
        Network accuracy per epoch
    """
    state = {
        'epoch': epoch,
        'description': description,
        'train_loss': losses[0],
        'val_loss': losses[1],
        'indices': indices,
        'state_dict': network.state_dict(),
        'optimiser': network.optimiser.state_dict(),
        'scheduler': network.scheduler.state_dict(),
    }

    if accuracy:
        state['accuracy'] = accuracy

    torch.save(state, save_path)
