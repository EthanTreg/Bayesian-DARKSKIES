"""
Classes for encoder, decoder, or autoencoder type architectures
"""
import torch
from torch import Tensor, nn
from netloader.network import Network

from src.utils.utils import label_change
from src.networks.base import BaseNetwork


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

    def _loss(self, high_dim: Tensor, low_dim: Tensor) -> float:
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
        float
            Loss from the autoencoder's predictions'
        """
        bounds = torch.tensor([0., 1.]).to(self._device)
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

        self._update(loss)
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
    """
    def _loss(self, high_dim: Tensor, low_dim: Tensor) -> float:
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
        float
            Loss from the network's predictions'
        """
        output = self.network(low_dim)
        loss = nn.MSELoss()(output, high_dim)
        self._update(loss)
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
    batch_predict(high_dim) -> Tensor
        Generates predictions for the given data
    """
    def __init__(
            self,
            save_num: int,
            states_dir: str,
            network: Network,
            description: str = '',
            classes: Tensor = None,
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
        classes : Tensor, default = None
            Unique classes if using class classification
        loss_function : Module, default = MSELoss
            Loss function to use
        """
        super().__init__(save_num, states_dir, network, description=description)
        self._classes = classes.to(self._device)
        self._loss_function = loss_function

    def _loss(self, high_dim: Tensor, low_dim: Tensor) -> float:
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
        float
            Loss from the network's predictions'
        """
        output = self.network(high_dim)

        # Default shape is (N, L), but cross entropy expects (N)
        if isinstance(self._loss_function, nn.CrossEntropyLoss):
            low_dim = label_change(low_dim.squeeze(), self._classes)

        loss = self._loss_function(output, low_dim)
        self._update(loss)
        return loss.item()

    def batch_predict(self, data: Tensor) -> Tensor:
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
        output = super().batch_predict(data)

        if isinstance(self._loss_function, nn.CrossEntropyLoss):
            output = torch.argmax(output, dim=-1, keepdim=True)

        return output
