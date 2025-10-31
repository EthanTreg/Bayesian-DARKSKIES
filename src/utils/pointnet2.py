"""
Implements PointNet++ from https://arxiv.org/pdf/1706.02413 with official GitHub
https://github.com/charlesq34/pointnet2
"""
from copy import deepcopy
from typing import Any, Literal, cast, overload

import torch
from torch import nn, Tensor
from netloader import layers
from netloader.utils import Shapes
from netloader.data import DataList
from netloader.network import Network
from netloader.utils.types import TensorListLike
from netloader.layers.base import BaseLayer, BaseSingleLayer


def farthest_point_sample(features: int, x: Tensor) -> Tensor:
    """
    Generalized Farthest Point Sampling (FPS).

    Parameters
    ----------
    features : int
        Number of points to sample
    x : Tensor
        Input points with shape (...,N,C) and type float, where N is the number of points and C is
        the number of input features

    Returns
    -------
    centroids : Tensor
        Indices of sampled points with shape (...,S) and type int, where S is the number of
        sampled points
    """
    i: int
    batch: int
    idxs: Tensor
    distance: Tensor
    farthest: Tensor
    centroids: Tensor
    device: torch.device = x.device
    shape: torch.Size = x.shape[:-2]

    x = x.view(-1, *x.shape[-2:])
    batch = len(x)
    idxs = torch.arange(batch, device=device)
    distance = torch.ones(x.shape[:-1], device=device) * 1e10
    farthest = torch.randint(0, x.size(-2), [batch], device=device)
    centroids = torch.zeros((batch, features), dtype=torch.long, device=device)

    for i in range(features):
        centroids[:, i] = farthest
        centroid = x[idxs, farthest].unsqueeze(dim=-2)
        dist = torch.sum((x - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids.view(*shape, features)


def index_points(x: Tensor, idxs: Tensor) -> Tensor:
    """
    Index into points with arbitrary batch dimensions.

    Parameters
    ----------
    x : Tensor
        Input points with shape (B#,N,C) and type float, where B# can be any number of dimensions, N
        is the number of points and C is the number of features
    idxs : Tensor
        Indices with shape (B#,S#) and type float, where S# is the number of samples for an
        arbitrary number of dimensions

    Returns
    -------
    Tensor
        Indexed points with shape (B#,S#,C) and type float
    """
    batch_idxs: Tensor
    device: torch.device = x.device
    shape: torch.size = idxs.shape

    x = x.view(-1, *x.shape[-2:])
    idxs = idxs.view(len(x), -1)
    batch_idxs = torch.arange(len(x), dtype=torch.long, device=device)[:, None]
    return x[batch_idxs, idxs].view(*shape, x.size(-1))


def query_ball_point(samples: int, radius: float, x: Tensor, new_x: Tensor) -> Tensor:
    """
    Find neighboring points within a ball query.

    Parameters
    ----------
    samples : int
        Maximum number, K, of points to sample in each local region
    radius : float
        Local region radius, points farther than this distance from each query point are excluded
    x : Tensor
        All points in the set with shape (...,N,C) and type float, where N is the number of points
        and C is the number of features
    new_x : Tensor
        Query points with shape (...,S,C) and type float, where S is the number of query points

    Returns
    -------
    group_idx : Tensor
        Indices of grouped points within the radius with shape (...,S,K) and type long, if there are
        fewer samples than K, then the first point is repeated to pad the indices
    """
    device: torch.device = x.device
    shape: torch.Size = x.shape[:-2]

    # Indices of all points
    x = x.view(-1, *x.shape[-2:])
    new_x = new_x.view(-1, *new_x.shape[-2:])
    group_idx = torch.arange(x.size(-2), device=device).repeat(len(x), new_x.size(-2), 1)

    # Mask out points outside radius
    group_idx[torch.cdist(new_x, x) > radius ** 2] = x.size(-2)

    # Take closest nsample
    group_idx = group_idx.sort(dim=-1)[0][..., :samples]

    # Deal with empty neighborhoods by repeating first
    group_first = group_idx[..., :1].expand(len(x), new_x.size(-2), group_idx.size(-1))
    mask = group_idx == x.size(-2)
    group_idx[mask] = group_first[mask]
    group_idx = torch.cat((
        group_idx,
        group_idx[..., :1].expand(*group_idx.shape[:-1], samples - group_idx.size(-1)),
    ), dim=-1)
    return group_idx.view(*shape, *group_idx.shape[-2:])


class Max(BaseLayer):
    """
    Take the maximum value along a specified dimension of a tensor or each tensor in a DataList.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x) -> TensorListLike
        Forward pass for taking the maximum value along a specified dimension of a tensor or each
        tensor in a DataList
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(self, dim: int, shapes: Shapes, *, keepdim: bool = False, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        dim : int
            Dimension to take the maximum over
        shapes : Shapes
            Shape of the outputs from each layer
        keepdim : bool, default = False
            Whether to retain the reduced dimension with size 1
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        elm: list[int]
        shape: list[int] | list[list[int]] = shapes.get(-1, list_=True).copy()
        self._dim: int = dim
        self._keepdim: bool = keepdim
        shapes.append([self.update_shape(elm) for elm in shape] if isinstance(shape[0], list) else
                      self.update_shape(shape))

    def update_shape(self, shape: list[int]) -> list[int]:
        """
        Update the shape after taking the maximum along a specified dimension.

        Parameters
        ----------
        shape : list[int]
            Input shape

        Returns
        -------
        list[int]
            Updated shape
        """
        shape = shape.copy()

        if self._keepdim:
            shape[self._dim] = 1
        else:
            shape.pop(self._dim)
        return shape

    def forward(self, x: TensorListLike, *_: Any, **__: Any) -> TensorListLike:
        """
        Forward pass for taking the maximum value along a specified dimension of a tensor or each
        tensor in a DataList.

        Parameters
        ----------
        x : TensorListLike
            Input with tensors of shape (N#,C,M#) and type float, N# can be any number of
            dimensions, C is the number of channels, and M# can be any number of dimensions, C is
            the dimension to take the maximum over and each tensor can have a different value for C

        Returns
        -------
        TensorListLike
            Output with tensors of shape (N#,1,M#) if keepdim is True else (N#,M#) and type float
        """
        if isinstance(x, DataList):
            return DataList([datum.max(dim=self._dim, keepdim=self._keepdim)[0] for datum in x])
        return x.max(dim=self._dim, keepdim=self._keepdim)[0]

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network.

        Returns
        -------
        str
            Layer parameters
        """
        return f'dim={self._dim}, keepdim={self._keepdim}'


class Permute(BaseLayer):
    """
    Permute the dimensions of a tensor or each tensor in a DataList.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x) -> TensorListLike
        Forward pass for permuting the dimensions of a tensor or each tensor in a DataList
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(self, perm: list[int], shapes: Shapes, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        perm : list[int]
            List of dimension indices to permute the input tensor to
        shapes : Shapes
            Shape of the outputs from each layer
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        i: int
        j: int
        elm: list[int]
        shape: list[int] | list[list[int]] = shapes.get(-1, list_=True).copy()
        self.perm: list[int] = perm

        if isinstance(shape[0], list):
            shape = cast(list[list[int]], shape)

            for i, elm in enumerate(shape):
                shape[i] = [elm[j] for j in self.perm]
        else:
            shape = [shape[i] for i in self.perm]

        shapes.append(shape)

    def forward(self, x: TensorListLike, *_: Any, **__: Any) -> TensorListLike:
        """
        Forward pass for permuting the dimensions of a tensor or each tensor in a DataList.

        Parameters
        ----------
        x : TensorListLike
            Input with tensors of shape (N,M#) and type float, N is the batch size and M# can be any
            number of dimensions to permute

        Returns
        -------
        TensorListLike
            Output with tensors of shape (N,M'#) and type float, where M'# is the permuted shape of
            M#
        """
        datum: Tensor

        if isinstance(x, DataList):
            return DataList([datum.permute(*self.perm) for datum in x])
        return x.permute(0, *self.perm)

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network.

        Returns
        -------
        str
            Layer parameters
        """
        return f'perm={self.perm}'


class DataListConcat(BaseLayer):
    """
    Concatenate a DataList into a single tensor.

    If input is a single tensor, it is returned unchanged.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x) -> Tensor
        Concatenate a DataList into a single tensor
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(self, dim: int, shapes: Shapes, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        dim : int
            Dimension to concatenate over
        shapes : Shapes
            Shape of the outputs from each layer
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._dim: int = dim
        shape: list[int] | list[list[int]] = shapes.get(-1, list_=True).copy()

        if isinstance(shape[0], list):
            shape = cast(list[list[int]], shape)

            for elm in shape[1:]:
                self._check_concatenation(elm, shape[0])

            new_shape = shape[0].copy()
            new_shape[self._dim] = sum(elm[self._dim] for elm in shape)
        else:
            new_shape = shape

        shapes.append(new_shape)

    def _check_concatenation(self, shape: list[int], target: list[int]) -> None:
        """
        Checks if input shape and target shape are compatible for concatenation.

        Parameters
        ----------
        shape : list[int]
            Input shape
        target : list[int]
            Target shape
        """
        if ((target[:self._dim] + (target[self._dim + 1:] if self._dim != -1 else []) !=
             shape[:self._dim] + (shape[self._dim + 1:] if self._dim != -1 else [])) or
                (len(target) != len(shape))):
            raise ValueError(f'Input shape {shape} does not match the target shape {target} for '
                             f'concatenation over dimension {self._dim}')

    def forward(self, x: TensorListLike, *_: Any, **__: Any) -> Tensor:
        """
        Forward pass for concatenating a DataList into a single tensor.

        Parameters
        ----------
        x : TensorListLike
            Input with tensors of shape (N#,C,M#) and type float, where N# can be any number of
            dimensions, C is the number of channels, and M# can be any number of dimensions, C is
            the dimension to concatenate over and each tensor can have a different value for C

        Returns
        -------
        Tensor
            Output with shape (N#,C',M#) and type float, where C' is the sum of the C
            dimensions from each input tensor
        """
        if isinstance(x, Tensor):
            return x
        return torch.cat([*x], dim=self._dim)

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network.

        Returns
        -------
        str
            Layer parameters
        """
        return f'dim={self._dim}'


class PointGroupAll(BaseLayer):
    """
    PointNet++ grouping layer that groups all points into a single region.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x) -> DataList[Tensor]
        Forward pass for the PointNet++ grouping layer
    """
    def __init__(self, shapes: Shapes, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        shapes : Shapes
            Shape of the outputs from each layer
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        shape = shapes[(-1, 0)].copy()
        shape[-3:-3] = [1]

        if len(shapes.get(-1, list_=True)) > 2:
            shapes.append([shape, shape[:-1] + [shapes[(-1, -1)][-1]]])
        else:
            shapes.append(shape)

    def forward(self, x: DataList[Tensor], *_: Any, **__: Any) -> DataList[Tensor]:
        """
        Forward pass for the PointNet++ group all layer.

        Parameters
        ----------
        x : DataList[Tensor]
            Input with tensors of shape (...,N,C), (...,S,C), and optionally (...,N,D) and type
            float, where N is the number of points, S is the number of sampled points, C is the
            number of input coordinates, and D is the number of input features, the first tensor is
            the input point coordinates, the second is the new point coordinates, and the third is
            the input point features, if provided

        Returns
        -------
        DataList[Tensor]
            Output with tensors of shape (...,N,1,C) and optionally (...,N,1,D) and type float,
            where the first tensor is the input point coordinates and the second is the input point
            features, if provided
        """
        if len(x) > 2:
            return DataList([
                x.get(0, list_=True).unsqueeze(dim=-3),
                x.get(-1, list_=True).unsqueeze(dim=-3),
            ])
        return DataList([x.get(0, list_=True).unsqueeze(dim=-3)])


class PointGroup(BaseLayer):
    """
    PointNet++ grouping layer

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x) -> DataList[Tensor]
        Forward pass for the PointNet++ grouping layer
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(self, samples: int, radius: float, shapes: Shapes, **kwargs: Any) -> None:
        """
        PointNet++ grouping layer.

        Parameters
        ----------
        samples : int
            Number of points to sample in each local region
        radius : float
            Local region radius, points farther than this distance from each query point are
            excluded
        shapes : Shapes
            Shape of the outputs from each layer
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._samples: int = samples
        self._radius: float = radius
        shape = shapes[(-1, 0)].copy()
        shape[-2] = samples
        shape[-2:-2] = shapes[(-1, 1)][-2:-1]

        if len(shapes.get(-1, list_=True)) > 2:
            shapes.append([shape, shape[:-1] + [shapes[(-1, -1)][-1]]])
        else:
            shapes.append(shape)

    def forward(self, x: DataList[Tensor], *_: Any, **__: Any) -> DataList[Tensor]:
        """
        Forward pass for the PointNet++ grouping layer.

        Parameters
        ----------
        x : DataList[Tensor]
            Input with tensors of shape (...,N,C), (...,S,C), and optionally (...,N,D) and type
            float, where N is the number of points, S is the number of sampled points, C is the
            number of input coordinates, and D is the number of input features, the first tensor
            must be the point coordinates, the second is the sampled point coordinates, and an
            optional third tensor with point features

        Returns
        -------
        DataList[Tensor]
            Output with tensors of shape (...,N,K,C), (...,S,K,C), and optionally (...,N,K,D) and
            type float, where K is the number of samples in each local region
        """
        group_idx = query_ball_point(self._samples, self._radius, *x.get(slice(2), list_=True))
        grouped_x = index_points(x.get(0, list_=True), group_idx)
        grouped_x -= x.get(1, list_=True).unsqueeze(dim=-2)

        if len(x) > 2:
            return DataList([grouped_x, index_points(x.get(2, list_=True), group_idx)])
        return DataList([grouped_x])

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network.

        Returns
        -------
        str
            Layer parameters
        """
        return f'samples={self._samples}, radius={self._radius}'


class PointSample(BaseLayer):
    """
    PointNet++ sampling layer.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x) -> DataList[Tensor]
        Forward pass of the PointNet++ sampling layer
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    @overload
    def __init__(self, shapes: Shapes, group_all: Literal[True], **kwargs: Any) -> None: ...

    @overload
    def __init__(self, shapes: Shapes, points: int, **kwargs: Any) -> None: ...

    def __init__(
            self,
            shapes: Shapes,
            group_all: bool = False,
            points: int = 1,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        shapes : Shapes
            Shape of the outputs from each layer
        group_all : bool, default = False
            If True, all points are grouped into a single region, points parameter is ignored
        points : int, default = 1
            Number of points to sample in the layer, if 0 all points are used, not used if group_all
            is True
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        shape: list[list[int]] = deepcopy(shapes.get(-1, True))
        self._group_all: bool = group_all
        self._points: int = points

        if shapes.check(-1):
            raise ValueError(f'PointSamples requires input to be a DataList and therefore input '
                             f'shape to be a list of shapes; however, input has shape: '
                             f'{shapes[-1]}')

        shape.insert(1, shape[0])
        shape[1][-2] = 1 if self._group_all else self._points
        shapes.append(shape)

    def forward(self, x: DataList[Tensor], *_: Any, **__: Any) -> DataList[Tensor]:
        """
        Forward pass of the PointNet++ sampling layer.

        Parameters
        ----------
        x : DataList[Tensor]
            Input with tensors of shape (...,N,C) and optionally (...,N,D) and type float, where N
            is the number of points and D is the number of input features, the first tensor must be
            the point coordinates with an optional second tensor with point features

        Returns
        -------
        DataList[Tensor]
            Output with tensors of shape (...,N,C), (...,S,C), and optionally (...,N,D) and type
            float, where S is the number of sampled points and D is the number of output features,
            the first tensor is the input point coordinates, the second is the new point
            coordinates, and the third is the input point features, if provided
        """
        x = x.copy()
        x_coords: Tensor = x.get(0, list_=True)

        if self._group_all:
            x.insert(1, torch.zeros(
                (*x_coords.shape[:-2], 1, x_coords.size(-1)),
                device=x_coords.device,
            ))
        else:
            x.insert(1, index_points(
                x_coords,
                farthest_point_sample(self._points, x_coords),
            ))
        return x

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network.

        Returns
        -------
        str
            Layer parameters
        """
        return f'group_all={self._group_all}, points={self._points}'


class PointNetSetAbstractionMsg(BaseSingleLayer):
    """
    PointNet++ set abstraction layer with multi-scale grouping

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : nn.Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    _group_all: bool = False

    def __init__(
            self,
            points: int,
            samples: list[int],
            radii: list[float],
            features: list[list[int]],
            shapes: Shapes,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        points : int
            Number of points to sample in the layer, if 0 all points are used, not used if group_all
            is True
        samples : list[int]
            List of number of samples for each scale in the layer, not used if group_all is True
        radii : list[float]
            List of radii for each scale in the layer, not used if group_all is True
        features : list[list[int]]
            List of features for each linear layer for each scale in the layer
        shapes : Shapes
            Shape of the outputs from each layer
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        sample: int
        idx: int | tuple[int, int] = -1 if isinstance(shapes.get(-1, list_=True)[0], int) else \
            (-1, 0)
        radius: float
        block_features: list[int]
        shape: list[list[int]]
        self._points: int = 1 if self._group_all else points or shapes[idx][0]
        self._samples: list[int] = samples
        self._radii: list[float] = radii
        self._features: list[list[int]] = features
        self.block_shapes: list[Shapes] = []
        self.layers: nn.ModuleList = nn.ModuleList()
        self.shapes: Shapes = shapes[-1:]

        self.layers.append(PointSample(self.shapes, group_all=self._group_all, points=self._points))

        for sample, radius, block_features in zip(self._samples, self._radii, self._features):
            self.shapes.append(self.shapes.get(1, True))
            self.layers.append(self.block(sample, radius, block_features))

        shape = [self.shapes[-1].copy(), self.shapes[-1].copy()]
        shape[0][-1] = self.shapes.get(0, True)[0][-1]
        shape[1][-1] = sum(block_features[-1] for block_features in features)
        shapes.append(shape)

    @staticmethod
    def network(features: list[int], shapes: Shapes) -> nn.Sequential:
        """
        Create the linear layers for each scale in the PointNet++ set abstraction layer.

        Parameters
        ----------
        features : list[int]
            List of features for each linear layer
        shapes : Shapes
            Network Shapes object to append the final layer shape to

        Returns
        -------
        nn.Sequential
            Linear layers
        """
        net_shapes: Shapes = Shapes([shapes[-1].copy()])
        net: nn.Sequential = nn.Sequential(*[layers.Conv(
            [],
            net_shapes,
            filters=feature,
            kernel=1,
            norm='batch',
        ) for feature in features])
        shapes.append(net_shapes[-1])
        return net

    def block(
            self,
            sample: int,
            radius: float,
            features: list[int]) -> nn.Sequential:
        """
        Create a single block for the PointNet++ set abstraction layer.

        Parameters
        ----------
        sample : int
            Number of points to sample in each local region
        radius : float
            Local region radius, points farther than this distance from each query point are
            excluded
        features : list[int]
            List of features for each linear layer

        Returns
        -------
        nn.Sequential
            Block layers
        """
        block_shapes: Shapes = Shapes([deepcopy(self.shapes.get(-1, list_=True))])
        block: nn.Sequential = nn.Sequential(
            PointGroupAll(block_shapes) if self._group_all else
            PointGroup(sample, radius, block_shapes),
            DataListConcat(-1, block_shapes),
            Permute(list(range(len(block_shapes[-1]) - 3)) + [-1, -2, -3], block_shapes),
            self.network(features, block_shapes),
            Permute(list(range(len(block_shapes[-1]) - 3)) + [-1, -2, -3], block_shapes),
            Max(-2, block_shapes),
        )
        self.shapes.append(block_shapes[-1])
        self.block_shapes.append(block_shapes)
        return block

    def sample(self, x: DataList[Tensor]) -> None:
        """
        Sample points from input point cloud.

        The input will be modified in place to add the sampled points as the second tensor with
        shape (...,S,C), where S is the number of sampled points and C is the number of input
        coordinates.

        Parameters
        ----------
        x : DataList[Tensor]
            Input with tensors of shape (...,N,C) and optionally (...,N,D) and type float, where N
            is the number of points and D is the number of input features, the first tensor must be
            the point coordinates with an optional second tensor with point features
        """
        x_coords: Tensor = x.get(0, list_=True)

        if self._group_all:
            x.insert(1, torch.zeros(
                (*x_coords.shape[:-2], 1, x_coords.size(-1)),
                device=x_coords.device,
            ))
        else:
            x.insert(1, index_points(
                x_coords,
                farthest_point_sample(self._points, x_coords),
            ))

    def forward(self, x: DataList[Tensor], *_: Any, **__: Any) -> DataList[Tensor]:
        """
        Forward pass for PointNet++ set abstraction layer for single and multi-scale grouping.

        Parameters
        ----------
        x : DataList[Tensor]
            Input with tensors of shape (...,N,C) and optionally (...,N,D) and type float, where N
            is the number of points and C is the number of input coordinates and D is the number of
            input features, the first tensor must be the point coordinates with an optional second
            tensor with point features

        Returns
        -------
        DataList[Tensor]
            Output with tensors of shape (...,S,C) and optionally (...,S,D') and type float, where S
            is the number of sampled points and D' is the number of output features, the first
            tensor is the new point coordinates and the second tensor is the new point features, if
            provided
        """
        new_points: list[Tensor] = []
        x = self.layers[0](x)

        for layer in self.layers[1:]:
            new_points.append(layer(x))
        return DataList([x.get(1, list_=True), torch.cat(new_points, dim=-1)])


class PointNetSetAbstraction(PointNetSetAbstractionMsg):
    """
    PointNet++ set abstraction layer

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            points: int,
            samples: int,
            radius: float,
            features: list[int],
            shapes: Shapes,
            *,
            group_all: bool = False,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        points : int
            Number of points to sample in the layer, if 0 all points are used
        samples : int
            Number of points to sample in each local region
        radius : float
            Local region radius, points farther than this distance from each query point are
            excluded
        features : list[int]
            List of features for each linear layer
        shapes : Shapes
            Shape of the outputs from each layer
        group_all : bool, default = False
            If True, all points are grouped into a single region, points, samples, and radius
            parameters are ignored
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        self._group_all = group_all
        super().__init__(
            points,
            [samples],
            [radius],
            [features],
            shapes,
            **kwargs,
        )


class PointNet2(Network):
    """
    PointNet++ network

    Attributes
    ----------
    layer_num : int
        Number of layers to use, if None use all layers
    group : int
        Which group is the active group if a layer has the group attribute
    name : str
        Name of the network, used for saving
    version : str
        NetLoader version string
    checkpoints : list[TensorListLike]
        Outputs from each checkpoint with each Tensor having shape (N,...) and type float, where N
        is the batch size
    config : dict[str, Any]
        Network configuration, not used in this network
    kl_loss : Tensor
        KL divergence loss on the latent space of shape (1) and type float, if using a sample layer
    net : nn.ModuleList
        Network construction
    check_shapes : Shapes
        Checkpoint output shapes
    shapes : Shapes
        Layer output shapes
    """
    def __init__(self, in_shape: list[int] | list[list[int]], out_shape: list[int]) -> None:
        """
        Parameters
        ----------
        in_shape : list[int] | list[list[int]]
            Input shape for the network
        out_shape : list[int]
            Output shape for the network
        """
        super().__init__(
            'pointnet2',
            {'net': {}, 'layers': []},
            in_shape,
            [],
            suppress_warning=True,
        )
        self._build_net(out_shape)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Sets the state of the network for pickling

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary containing the state of the network
        """
        super().__init__(
            state['name'],
            {'net': {}, 'layers': []},
            state['shapes'][0],
            [],
            suppress_warning=True,
        )
        self.shapes = Shapes([state['shapes'][0]])

        self._build_net(state['shapes'][-1])
        self.load_state_dict(state['net'])

    def _build_net(self, out_shape: list[int]) -> None:
        """
        Build the PointNet++ network.

        Parameters
        ----------
        out_shape : list[int]
            List of output shape for the network
        """
        self.net = nn.ModuleList([
            layers.Pack(False, shapes=self.shapes, check_shapes=Shapes()),
            PointNetSetAbstractionMsg(
                512,
                [16, 32, 128],
                [0.1, 0.2, 0.4],
                [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                self.shapes,
            ),
            PointNetSetAbstractionMsg(
                128,
                [32, 64, 128],
                [0.2, 0.4, 0.8],
                [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                self.shapes,
            ),
            PointNetSetAbstraction(
                0,
                0,
                0,
                [256, 512, 1024],
                self.shapes,
                group_all=True,
            ),
            *self._head(out_shape)
        ])

    def _head(self, net_out: list[int]) -> list[BaseLayer]:
        """
        Creates the head of the PointNet++ network.

        Parameters
        ----------
        net_out : list[int]
            List of output shape for the network

        Returns
        -------
        list[BaseLayer]
            Head of the network
        """
        return [
            layers.Index(idx=1, map_=False, shapes=self.shapes),
            layers.Linear(net_out, self.shapes, features=512, batch_norm=True, dropout=0.4),
            layers.Linear(net_out, self.shapes, features=256, batch_norm=True, dropout=0.5),
            layers.Linear(net_out, self.shapes, factor=1, activation=None),
        ]

    def forward(self, x: DataList[Tensor]) -> TensorListLike:
        """
        Forward pass for the PointNet++ network.

        Parameters
        ----------
        x : DataList[Tensor]
            Input with tensors of shape (...,N,C) and optionally (...,N,D) and type float, where N
            is the number of points and C is the number of input coordinates and D is the number of
            input features, the first tensor must be the point coordinates with an optional second
            tensor with point features

        Returns
        -------
        TensorListLike
            Output with shape (N,C') and type float, where C' is the number of output features
        """
        return super().forward(x)
