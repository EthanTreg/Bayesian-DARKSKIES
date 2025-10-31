"""
ConvNeXtCluster model with clustering capabilities.
"""
from typing import Any

from netloader import layers
from netloader.utils import Shapes
from netloader.models import convnext
from netloader.layers.base import BaseLayer

from src.utils.pointnet2 import PointNet2


class ConvNeXtCluster(convnext.ConvNeXtTiny):
    """
    A ConvNeXtTiny network with clustering capabilities.
    Inherits from ConvNeXtTiny and adds clustering functionality.
    """
    def __init__(
            self,
            in_shape: list[int],
            out_shape: list[int],
            latent_dim: int = 7,
            max_drop_path: float = 0,
            layer_scale: float = 1e-6) -> None:
        """
        Parameters
        ----------
        in_shape : list[int] | list[list[int]] | tuple[int, ...]
            Shape of the input tensor(s), excluding batch size
        out_shape : list[int]
            shape of the output tensor, excluding batch size
        latent_dim : int, default = 7
            Dimensionality of the latent space
        max_drop_path : float, default = 0
            Maximum drop path fraction
        layer_scale : float, default = 1e-6
            Default value for learnable parameters to scale the convolutional filters in a ConvNeXt
            block
        """
        self._latent_dim: int = latent_dim
        super().__init__(
            in_shape,
            out_shape,
            max_drop_path=max_drop_path,
            layer_scale=layer_scale,
        )

    def __getstate__(self) -> dict[str, Any]:
        """
        Returns a dictionary containing the state of the network for pickling

        Returns
        -------
        dict[str, Any]
            Dictionary containing the state of the network
        """
        return super().__getstate__() | {'latent_dim': self._latent_dim}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Sets the state of the network for pickling

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary containing the state of the network
        """
        self._latent_dim = state['latent_dim'] if 'latent_dim' in state else 7
        super().__setstate__(state)

    def _head(self, out_shape: list[int], **kwargs: Any) -> list[BaseLayer]:
        """
        Head of ConvNeXt for adapting the learned features into the desired output.

        Parameters
        ----------
        out_shape : list[int]
            shape of the output tensor, excluding batch size

        **kwargs
            Global network parameters

        Returns
        -------
        list[BaseLayer]
            Layers used in the head
        """
        return [
            layers.AdaptivePool(1, self.shapes, **kwargs),
            layers.Reshape([-1], shapes=self.shapes, **kwargs),
            layers.LayerNorm(dims=1, shapes=self.shapes, **kwargs),
            layers.Linear(out_shape, self.shapes, features=self._latent_dim, **kwargs),
            layers.OrderedBottleneck(self.shapes, min_size=1, **kwargs),
            layers.Checkpoint(self.shapes, **kwargs),
            layers.Linear(out_shape, self.shapes, factor=1, activation=None, **kwargs),
        ]


class PointNet2Cluster(PointNet2):
    """
    A PointNet2 network with clustering capabilities.
    Inherits from PointNet2 and adds clustering functionality.
    """
    def __init__(
            self,
            in_shape: list[int],
            out_shape: list[int],
            latent_dim: int = 7) -> None:
        """
        Parameters
        ----------
        in_shape : list[int] | list[list[int]] | tuple[int, ...]
            Shape of the input tensor(s), excluding batch size
        out_shape : list[int]
            shape of the output tensor, excluding batch size
        latent_dim : int, default = 7
            Dimensionality of the latent space
        """
        self._latent_dim: int = latent_dim
        super().__init__(in_shape, out_shape)

    def _head(self, net_out: list[int]) -> list[BaseLayer]:
        """
        Constructs the head of the PointNet2 network.

        Parameters
        ----------
        net_out : list[int]
            Shape of the output tensor, excluding batch size

        Returns
        -------
        list[BaseLayer]
            Head of the PointNet2 network
        """
        return [
            layers.AdaptivePool(1, self.shapes),
            layers.Reshape([-1], shapes=self.shapes),
            layers.LayerNorm(dims=1, shapes=self.shapes),
            layers.Linear(net_out, self.shapes, features=self._latent_dim),
            layers.OrderedBottleneck(self.shapes, min_size=1),
            layers.Checkpoint(self.shapes, Shapes()),
            layers.Linear(net_out, self.shapes, factor=1, activation=None),
        ]
