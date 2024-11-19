"""
Architectures that cluster data
"""
from typing import Any, Self

import torch
import numpy as np
from numpy import ndarray
from torch import Tensor, nn
from torch.utils.data import DataLoader
from netloader.networks import BaseNetwork
from netloader.utils.utils import label_change
from netloader.transforms import BaseTransform
from netloader.layers.utils import BaseLayer
from netloader.network import Network
from netloader import layers


class CosineSimilarity(BaseNetwork):
    """
    Maximises the similarity/minimises latent space distance between inputs of the same class and
    maximises the difference/minimises latent space distance between inputs of different classes

    Attributes
    ----------
    save_path : str
        Path to the network save file
    optimiser : Optimizer
        Network optimiser, uses AdamW optimiser
    scheduler : LRScheduler
        Optimiser scheduler, uses reduce learning rate on plateau
    net : Module | Network
        Neural network
    margin : float, default = 0
        Cosine threshold for similarity to be non-zero
    description : str, default = ''
        Description of the network
    losses : tuple[list[float], list[float]], default = ([], [])
        Network training and validation losses
    header : dict[str, BaseTransform | None], default = {...: None, ...}
        Keys for the output data from predict and corresponding transforms
    idxs: (N) ndarray, default = None
        Data indices for random training & validation datasets
    in_transform : BaseTransform, default = None
        Transformation for the input data
    """
    def __init__(
            self,
            save_num: int,
            states_dir: str,
            net: nn.Module | Network,
            mix_precision: bool = False,
            learning_rate: float = 1e-3,
            description: str = '',
            verbose: str = 'full',
            transform: BaseTransform | None = None,
            in_transform: BaseTransform | None = None):
        super().__init__(
            save_num,
            states_dir,
            net,
            mix_precision=mix_precision,
            learning_rate=learning_rate,
            description=description,
            verbose=verbose,
            transform=transform,
            in_transform=in_transform,
        )
        self.margin: float = 0
        self.header['preds'] = None

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {'margin': self.margin}

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.margin = state['margin']

    def _loss(self, in_data: Tensor, target: Tensor) -> float:
        """
        Calculates the loss from the cosine similarity

        Parameters
        ----------
        in_data : (N,...) Tensor
            Input data of batch size N and the remaining dimensions depend on the network used
        target : (N,1) Tensor
            Class label for each output in the batch of size N

        Returns
        -------
        float
            Loss
        """
        output: Tensor = self.net(in_data)
        loss: Tensor = nn.CosineSimilarity(dim=-1)(output[None], output[:, None])
        idxs: Tensor

        target = target.squeeze()
        idxs = target[None] == target[:, None]
        loss[idxs] = 1 - loss[idxs]
        loss[~idxs] = torch.maximum(loss.new_tensor(0), loss[~idxs] - self.margin)
        loss = torch.mean(loss)
        self._update(loss)
        return loss.item()


class ClusterEncoder(BaseNetwork):
    """
    Clusters data into defined class and unknown classes based on cluster compactness, data distance
    in feature space and latent space, and distance between cluster centers and label distance for
    known classes

    It is assumed that the smallest labels are unknown up to the provided value for unknown

    It is assumed that the third to last checkpoint is the feature space, the last checkpoint is
    the cluster latent space and the network output is the classification

    Attributes
    ----------
    save_path : str
        Path to the network save file
    classes : (C) Tensor
        Classes of size C for clustering
    optimiser : Optimizer
        Network optimiser, uses AdamW optimiser
    scheduler : LRScheduler
        Optimiser scheduler, uses reduce learning rate on plateau
    net : Module | Network
        Neural network
    sim_loss : float, default = 1
        Loss weight for the difference between the distance in feature space and latent space
    compact_loss : float, default = 1
        Loss weight for the compactness of the cluster
    distance_loss : float, default = 1
        Loss weight for the distance between cluster centers for known classes
    class_loss : float, default = 1
        Loss weight for the class classification
    center_step : float, default = 1
        How far the cluster center should move towards the batch cluster center, if 0, cluster
        centers will be fixed
    description : str, default = ''
        Description of the network training
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current network training and validation losses
    header : dict[str, BaseTransform | None], default = {...: None, ...}
        Keys for the output data from predict and corresponding transforms
    idxs: (N) ndarray, default = None
        Data indices for random training & validation datasets
    in_transform : BaseTransform, default = None
        Transformation for the input data

    Methods
    -------
    batch_predict(data) -> tuple[Tensor, Tensor, Tensor]
        Generates predictions and latent space for the given data batch
    """
    def __init__(
            self,
            save_num: int,
            states_dir: str,
            classes: Tensor,
            net: nn.Module | Network,
            mix_precision: bool = False,
            unknown: int = 1,
            learning_rate: float = 1e-3,
            description: str = '',
            method: str = 'mean',
            verbose: str = 'full',
            transform: BaseTransform | None = None,
            in_transform: BaseTransform | None = None):
        """
        Parameters
        ----------
        save_num : int
            File number to save the network
        states_dir : str
            Directory to save the network
        classes : (C) Tensor
            Classes of size C for clustering
        net : Module | Network
            Network to predict low-dimensional data
        mix_precision: bool, default = False
            If mixed precision should be used
        unknown : int, default = 1
            Number of unknown classes
        learning_rate : float, default = 1e-3
            Optimiser initial learning rate, if None, no optimiser or scheduler will be set
        method : str, default = 'mean'
            Whether to calculate the center of a cluster using the 'mean' or 'median'
        description : str, default = ''
            Description of the network training
        verbose : {'epoch', 'full', 'progress', None}
            If details about each epoch should be printed ('epoch'), details about epoch and epoch
            progress (full), just total progress ('progress'), or nothing (None)
        transform : BaseTransform, default = None
            Transformation of the network's output
        in_transform : BaseTransform, default = None
            Transformation for the input data
        """
        net.scale = nn.Parameter(torch.tensor((1.,), requires_grad=True))
        super().__init__(
            save_num,
            states_dir,
            net,
            mix_precision=mix_precision,
            learning_rate=learning_rate,
            description=description,
            verbose=verbose,
            transform=transform,
            in_transform=in_transform,
        )
        self._unknown: int = unknown
        self._method: str = method
        self._cluster_centers: Tensor | None = None
        self.sim_loss: float = 0.7
        self.class_loss: float = 0.2
        self.compact_loss: float = 0.5
        self.distance_loss: float = 3
        self.classes: Tensor = classes
        self.center_step: Tensor = torch.ones(len(self.classes), device=self._device)

        self.header |= {'probs': None, 'latent': None}

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'unknown': self._unknown,
            'method': self._method,
            'cluster_centers': self._cluster_centers,
            'sim_loss': self.sim_loss,
            'class_loss': self.class_loss,
            'compact_loss': self.compact_loss,
            'distance_loss': self.distance_loss,
            'classes': self.classes,
            'center_step': self.center_step,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._unknown = state['unknown']
        self._method = state['method']
        self._cluster_centers = state['cluster_centers']
        self.sim_loss = state['sim_loss']
        self.class_loss = state['class_loss']
        self.compact_loss = state['compact_loss']
        self.distance_loss = state['distance_loss']
        self.classes = state['classes']
        self.center_step = state['center_step']

    def _init_clusters(self, latent_dim: int) -> None:
        """
        Initialises cluster centers based on provided class values

        Parameters
        ----------
        latent_dim : int
            Dimension of the latent space
        """
        self._cluster_centers = torch.zeros((len(self.classes), latent_dim)).to(self._device)
        self._cluster_centers[:, 0] = self.classes
        self._cluster_centers[:self._unknown, 0] = 0.5

    def _cluster_centers_loss(self, latent: Tensor, labels: Tensor) -> Tensor:
        """
        Updates the cluster centers for each class and calculates the standard deviation to the
        nearest cluster for each cluster

        Parameters
        ----------
        latent : (N,Z) Tensor
            Cluster latent space for batch size N and latent size of Z
        labels : (N) Tensor
            Class label for each latent point in batch size N

        Returns
        -------
        Tensor
            Loss for the cluster standard deviation in the direction to the nearest cluster
        """
        batch_class: Tensor
        loss: Tensor = torch.tensor(0.).to(self._device)

        # Calculate cluster centers for each class in the batch
        for batch_class in torch.unique(labels)[self._unknown:]:
            # Shift class cluster center towards class batch center
            if self._train_state and torch.any(self.center_step):
                self._update_centers(batch_class, latent, labels)

            if self.compact_loss and len(latent[labels == batch_class]) > 1:
                loss += self._cluster_loss(latent, labels, batch_class)

        return loss / (len(torch.unique(labels)) - self._unknown)

    def _cluster_loss(self, latent: Tensor, labels: Tensor, cluster_class: Tensor) -> Tensor:
        """
        Calculates the loss for the standard deviation to the nearest cluster for the given cluster

        Parameters
        ----------
        latent : (N,Z) Tensor
            Cluster latent space for batch size N and latent size of Z
        labels : (N) Tensor
            Class label for each latent point in batch size N
        cluster_class : Tensor
            Class value for the cluster to calculate the loss for

        Returns
        -------
        Tensor
            Loss for the cluster standard deviation in the direction to the nearest cluster
        """
        direcs: Tensor
        idxs: Tensor = self.classes == cluster_class
        class_vecs: Tensor

        assert isinstance(self._cluster_centers, Tensor)
        class_vecs = latent[labels == cluster_class] - self._cluster_centers[idxs]

        # Direction to the other classes
        direcs = self._cluster_centers[~idxs][self._unknown:] - self._cluster_centers[idxs]

        # Average scatter in the direction of the other classes
        return torch.mean(class_vecs @ direcs.T / torch.linalg.norm(direcs, dim=-1) ** 2)

    def _distance_loss(self) -> Tensor:
        """
        Calculates the loss for the difference in distance between the classes and class cluster
        centers for known classes

        Returns
        -------
        Tensor
            Loss for the difference in distance for each class
        """
        centers: Tensor
        known_classes: Tensor = self.classes[self._unknown:]

        if len(known_classes) > 1:
            centers = self.net.scale * self._cluster_centers[self._unknown:, 0]
            return nn.MSELoss()(centers, known_classes)

        return torch.tensor(0., device=self._device)

    def _update_centers(self, batch_class: Tensor, latent: Tensor, labels: Tensor) -> None:
        """
        Shifts the cluster center for a class based on the batch and step size

        Parameters
        ----------
        batch_class : (1) Tensor
            Class value to shift the cluster center for
        latent : (N,Z) Tensor
            Cluster latent space for batch size N and latent size of Z
        labels : (N) Tensor
            Class label for each latent point in batch size N
        """
        batch_center: Tensor
        class_idx: Tensor = self.classes == batch_class
        label_idxs: Tensor = labels == batch_class

        if self._method == 'mean':
            batch_center = torch.mean(latent[label_idxs], dim=0)
        else:
            batch_center = torch.median(latent[label_idxs], dim=0)[0]

        self._cluster_centers[class_idx] += (
                (batch_center - self._cluster_centers[class_idx]) *
                self.center_step[class_idx]
        )

    def _loss(self, in_data: Tensor, target: Tensor) -> float:
        """
        Calculates the loss from the clusters

        Parameters
        ----------
        in_data : (N,...) Tensor
            Input high dimensional data of batch size N and the remaining dimensions depend on the
            network used
        target : (N,1) Tensor
            Class label for each output in the batch of size N

        Returns
        -------
        float
            Loss for the batch
        """
        idxs: Tensor
        one_hot: Tensor
        loss: Tensor = torch.tensor(0.).to(self._device)
        output: Tensor = self.net(in_data)
        latent: Tensor = self.net.checkpoints[-1]

        if self._cluster_centers is None:
            self._init_clusters(latent.size(1))

        # Similarity loss between distances in feature space and distances in latent space
        if self.sim_loss and torch.isin(target, self.classes[:self._unknown]).any():
            idxs = torch.isin(target.flatten(), self.classes[:self._unknown])
            loss += self.sim_loss * nn.MSELoss()(
                torch.cdist(latent[idxs, :1], latent[~idxs, :1]) / np.sqrt(latent.size(-1)),
                torch.cdist(
                    self.net.checkpoints[-3][idxs],
                    self.net.checkpoints[-3][~idxs],
                ) / np.sqrt(self.net.checkpoints[-3].size(-1)),
            )

        # Shift class centers and calculate compactness loss
        if self.compact_loss or self.distance_loss:
            loss += self.compact_loss * self._cluster_centers_loss(latent, target.squeeze())

        # Loss for the difference in class values and cluster distances
        if self.distance_loss:
            loss += self.distance_loss * self._distance_loss()

        # Classification loss
        if self.class_loss:
            one_hot = label_change(target.squeeze(), self.classes, one_hot=True)
            loss += self.class_loss * nn.CrossEntropyLoss()(output, one_hot)

        # Update network
        self._update(loss)

        # Prevent memory leak from saving graphs generated by each batch
        self._cluster_centers = self._cluster_centers.detach()
        return loss.item()

    def batch_predict(self, data: Tensor, **_: Any) -> tuple[ndarray, ndarray, ndarray]:
        """
        Generates predictions and latent space for the given data batch

        Parameters
        ----------
        data : (N,...) Tensor
            N data to generate predictions for

        Returns
        -------
        tuple[(N) ndarray, (N,C) ndarray, (N,Z) ndarray]
            N predictions, prediction probabilities for C classes and latent space points of
            dimension Z for the given data
        """
        probs: Tensor = self.net(data)
        predicts: Tensor = label_change(
            torch.argmax(probs, dim=1),
            torch.arange(probs.size(1)).to(self._device),
            out_label=self.classes,
        )
        return (
            predicts.detach().cpu().numpy(),
            probs.detach().cpu().numpy(),
            self.net.checkpoints[-1].detach().cpu().numpy(),
        )

    def to(self, *args: Any, **kwargs: Any) -> Self:  # pylint: disable=missing-function-docstring
        super().to(*args, **kwargs)
        self.classes = self.classes.to(*args, **kwargs)
        self.center_step = self.center_step.to(*args, **kwargs)

        if self._cluster_centers is not None:
            self._cluster_centers = self._cluster_centers.to(*args, **kwargs)

        return self


class CompactClusterEncoder(ClusterEncoder):
    """
    Semi supervised clustering of the latent space and prediction of labels

    Attributes
    ----------
    classes : (C) Tensor
        Classes of size C for clustering
    optimiser : Optimizer
        Network optimiser, uses AdamW optimiser
    scheduler : LRScheduler
        Optimiser scheduler, uses reduce learning rate on plateau
    net : Module | Network
        Encoder network
    cluster_loss : float, default = 1
        Weighting of the cluster loss
    distance_loss : float, default = 1
        Loss weight for the distance between cluster centers for known classes
    class_loss : float, default = 1
        Weighting of the cross entropy loss
    center_step : float, default = 1
        How far the cluster center should move towards the batch cluster center, if 0, cluster
        centers will be fixed
    description : str, default = ''
        Description of the network training
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current encoder training and validation losses
    header : dict[str, BaseTransform | None], default = {...: None, ...}
        Keys for the output data from predict and corresponding transforms
    idxs: (N) ndarray, default = None
        Data indices for random training & validation datasets
    in_transform : BaseTransform, default = None
        Transformation for the input data
    """
    def __init__(
            self,
            save_num: int,
            states_dir: str,
            classes: Tensor,
            net: nn.Module | Network,
            mix_precision: bool = False,
            steps: int = 3,
            unknown: int = 1,
            learning_rate: float = 1e-3,
            method: str = 'mean',
            description: str = '',
            verbose: str = 'full',
            transform: BaseTransform | None = None,
            in_transform: BaseTransform | None = None):
        """
        Parameters
        ----------
        save_num : int
            File number to save the network
        states_dir : str
            Directory to save the network
        classes : C Tensor
            Classes available for clustering for C possible classes
        net : Module | Network
            Network to cluster predict low-dimensional data
        mix_precision: bool, default = False
            If mixed precision should be used
        steps : int, default = 3
            Number of Markov chain steps for cluster identification
        unknown : int, default = 1
            Number of unknown classes
        learning_rate : float, default = 1e-3
            Optimiser initial learning rate, if None, no optimiser or scheduler will be set
        method : str, default = 'mean'
            Whether to calculate the center of a cluster using the 'mean' or 'median'
        description : str, default = ''
            Description of the network training
        verbose : {'epoch', 'full', 'progress', None}
            If details about each epoch should be printed ('epoch'), details about epoch and epoch
            progress (full), just total progress ('progress'), or nothing (None)
        transform : BaseTransform, default = None
            Transformation of the network's output
        in_transform : BaseTransform, default = None
            Transformation for the input data
        """
        super().__init__(
            save_num,
            states_dir,
            classes,
            net,
            mix_precision=mix_precision,
            unknown=unknown,
            learning_rate=learning_rate,
            method=method,
            description=description,
            verbose=verbose,
            transform=transform,
            in_transform=in_transform,
        )
        self._steps: int = steps
        self.sim_loss: float = 0  # Unused
        self.compact_loss: float = 0  # Unused
        self.class_loss: float = 0.2
        self.cluster_loss: float = 2.2
        self.distance_loss: float = 1

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {'steps': self._steps, 'cluster_loss': self.cluster_loss}

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._steps = state['steps']
        self.cluster_loss = state['cluster_loss']

    def _label_propagation_cluster_loss(self, latent: Tensor, one_hot: Tensor) -> Tensor:
        """
        Calculates the label propagation for unlabelled data points and calculates the loss for the
        clusters

        Parameters
        ----------
        latent : (N,Z) Tensor
            Cluster latent space for batch size N and latent size of Z
        one_hot : (L,C) Tensor
            One hot labels for L known datapoints and C classes

        Returns
        -------
        Tensor
            Cluster loss
        """
        step: int
        adjacency: Tensor
        posterior: Tensor
        agreement: Tensor
        transition: Tensor
        posterior_u: Tensor
        transition_uu: Tensor
        transition_ul: Tensor
        step_transition: Tensor
        masked_transition: Tensor
        optimal_transition: Tensor
        loss: Tensor = torch.tensor(0.).to(self._device)

        # Adjacency and transition matrices, softmax used instead of exponential and row
        # normalisation for stability
        adjacency = torch.matmul(latent, torch.t(latent))
        transition = nn.Softmax(dim=-1)(adjacency)
        transition = 1e-5 / len(transition) + (1 - 1e-5) * transition

        # Posterior calculation for label propagation
        if len(one_hot) < len(transition):
            transition_uu = transition[len(one_hot):, len(one_hot):]
            transition_ul = transition[len(one_hot):, :len(one_hot)]
            posterior_u = torch.matmul(torch.matmul(
                torch.linalg.inv(torch.eye(len(transition_uu)).to(self._device) - transition_uu),
                transition_ul,
            ), one_hot)
            posterior = torch.cat((one_hot, posterior_u))
        else:
            posterior = one_hot

        # Ideal compact clusters
        optimal_transition = torch.matmul(
            posterior,
            torch.t(posterior / (torch.sum(posterior, dim=0) + 1e-5)),
        )
        agreement = torch.matmul(posterior, torch.t(posterior))
        masked_transition = transition * agreement
        step_transition = transition

        # Markov chain for losses at different length graph chains
        for step in range(self._steps):
            if step:
                step_transition = torch.matmul(masked_transition, step_transition)

            loss -= torch.mean(
                optimal_transition * torch.log(step_transition),
            ) / self._steps

        return loss

    def _cluster_centers_loss(self, latent: Tensor, labels: Tensor) -> Tensor:
        """
        Calculates the centers of each class in the batch along the first dimension and the distance
        to the class value for known classes.

        Parameters
        ----------
        latent : (N,Z) Tensor
            Cluster latent space for batch size N and latent size of Z
        labels : (N) Tensor
            Class label for each latent point in batch size N

        Returns
        -------
        Tensor
            Loss for cluster centers in the batch
        """
        batch_centers: list[Tensor] = []
        label_idxs: Tensor
        batch_class: Tensor
        known_classes: Tensor = self.classes[self._unknown:]
        batch_classes: Tensor = known_classes[torch.isin(known_classes, torch.unique(labels))]

        if len(known_classes) == 0:
            return torch.tensor(0.).to(self._device)

        # Calculate cluster centers for each class in the batch
        for batch_class in batch_classes:
            label_idxs = labels == batch_class

            if self._method == 'mean':
                batch_centers.append(torch.mean(latent[label_idxs, 0]))
            else:
                batch_centers.append(torch.median(latent[label_idxs, 0]))

        return nn.MSELoss()(torch.stack(batch_centers), batch_classes)

    def _loss(self, in_data: Tensor, target: Tensor) -> float:
        """
        Calculates the loss from the network's predictions and clusters

        Parameters
        ----------
        in_data : (N,...) Tensor
            Input high dimensional data of batch size N and the remaining dimensions depend on the
            network used
        target : (N,1) Tensor
            Class label for each output in the batch of size N

        Returns
        -------
        float
            Loss for the network's predictions
        """
        output: Tensor = self.net(in_data)
        latent: Tensor = self.net.checkpoints[-1]

        loss = self._loss_function(target, latent, output)
        self._update(loss)
        return loss.item()

    def _loss_function(self, target: Tensor, latent: Tensor, output: Tensor) -> Tensor:
        """
        Loss function of the CompactClusterEncoder

        Parameters
        ----------
        target : (N,1) Tensor
            N target labels
        latent : (N,Z) Tensor
            N predicted 7D latent values
        output : (N,C) Tensor
            Predicted label scores

        Returns
        -------
        (1) Tensor
            Final loss
        """
        l_idxs: Tensor
        one_hot: Tensor
        bottleneck: Tensor
        loss: Tensor = torch.tensor(0.).to(self._device)

        target = target.squeeze()
        l_idxs = torch.isin(target, self.classes)

        # Find size of the bottleneck
        bottleneck = torch.argwhere(
            torch.count_nonzero(latent == 0, dim=0) == latent.size(0),
        ).flatten()

        # Remove part of latent space zeroed out due to information-ordered bottleneck layer
        if len(bottleneck) != 0:
            latent = latent[:, :bottleneck[0]]

        one_hot = label_change(target[l_idxs], self.classes, one_hot=True).float()

        # Classification loss
        if self.class_loss:
            loss += self.class_loss * nn.CrossEntropyLoss()(output[l_idxs], one_hot)

        # Cluster loss
        if self.cluster_loss:
            loss += self.cluster_loss * self._label_propagation_cluster_loss(
                torch.cat((latent[l_idxs], latent[~l_idxs])),
                one_hot,
            )

        # Distance loss
        if self.distance_loss:
            loss += self.distance_loss * self._cluster_centers_loss(latent[l_idxs], target[l_idxs])

        return loss

    def saliency(
            self,
            loader: DataLoader,
            batch: bool = True) -> dict[str, ndarray]:
        """
        Calculates the saliency for each dimension in the latent space by zeroing out the gradients

        Parameters
        ----------
        loader : DataLoader
            Dataset to calculate the saliencies for
        batch : bool, default = True
            If the saliencies should be calculated for the whole dataset or just one batch

        Returns
        -------
        dict[str, (N,...) ndarray]
            Prediction IDs, inputs, and saliencies
        """
        i: int
        idx: int
        saliencies: list[ndarray]
        data: dict[str, list[ndarray]] = {'ids': [], 'inputs': [], 'saliencies': []}
        output: Tensor
        latent: Tensor
        target: Tensor
        in_data: Tensor
        low_dim: Tensor
        high_dim: Tensor
        module: BaseLayer
        torch.backends.cudnn.enabled = False
        self.train(False)

        # Find the last checkpoint corresponding to the latent space
        for idx, module in enumerate(self.net.net[::-1]):
            if isinstance(module, layers.Checkpoint):
                idx = len(self.net.net) - 1 - idx
                break

        # Add gradient gate for saliency per latent dimension
        self.net.net.insert(idx, GradGate())
        self.net.layers.insert(idx, {'type': GradGate.__name__})

        # Calculate saliencies for the dataset
        for ids, low_dim, high_dim, *_ in loader:
            saliencies = []
            in_data, target = self._data_loader_translation(low_dim, high_dim)
            in_data = in_data.to(self._device)
            target = target.to(self._device)

            # Require gradients for the input and generate predictions
            in_data.requires_grad_()
            output = self.net(in_data)
            latent = self.net.checkpoints[-1]

            # Calculate saliency through backpropagation for each latent dimension
            for i in range(latent.size(-1)):
                self.net.net[idx].idx = i
                self._loss_function(target, latent, output).backward(retain_graph=True)
                saliencies.append(in_data.grad.data.cpu().numpy().copy())
                self.net.net.zero_grad()
                in_data.grad.zero_()

            data['ids'].append(ids)
            data['inputs'].append(in_data.detach().cpu().numpy())
            data['saliencies'].append(np.stack(saliencies, axis=1))

            # If calculating saliencies for only one batch
            if batch:
                break

        # Remove gradient gate
        del self.net.net[idx]
        del self.net.layers[idx]
        torch.backends.cudnn.enabled = True
        return {key: np.concatenate(value) for key, value in data.items()}


class GradGate(BaseLayer):
    """
    Zeros out all gradients in the backwards pass except one index in the last dimension

    Attributes
    ----------
    idx : int, default = 0
        Index to keep the gradient in the last dimension

    Methods
    -------
    forward(x) -> Tensor
        Forward pass to register the gradient hook
    hook(grad) -> Tensor
        Adds the gradient gate to the gradient in the backward pass
    """
    # type: ignore[annotation-unchecked]
    def __init__(self):
        super().__init__(idx=0)
        self.idx: int = 0

    def forward(self, x: Tensor, **_: Any) -> Tensor:
        """
        Forward pass to register the gradient hook

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor with batch size N

        Returns
        -------
        (N,...) Tensor
            Output tensor with batch size N
        """
        x.register_hook(self.hook)
        return x

    def hook(self, grad: Tensor) -> Tensor:
        """
        Adds the gradient gate to the gradient in the backward pass

        Parameters
        ----------
        grad : (N,...) Tensor
            Gradient from backpropagation of batch size N

        Returns
        -------
        (N,...) Tensor
            Gradient zeroed out except for the given index in the last dimension
        """
        gate: Tensor = torch.zeros(grad.size(-1)).to(grad.device)
        gate[self.idx] = 1
        return grad * gate
