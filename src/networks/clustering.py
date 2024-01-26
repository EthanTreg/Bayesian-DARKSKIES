"""
Architectures that cluster data
"""
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from netloader.network import Network

from src.networks.base import BaseNetwork


class ClusterEncoder(BaseNetwork):
    """
    Clusters data into defined class and unknown classes based on cluster compactness, data distance
    in feature space and latent space, and distance between cluster centers and label distance for
    known classes

    It is assumed that the smallest labels are unknown up to the provided value for unknown

    Attributes
    ----------
    classification : boolean, default = True
        Whether the network should be trained as a classifier or to cluster
    sim_loss : float, default = 1
        Loss for the difference between the distance in feature space and latent space
    compact_loss : float, default = 1
        Loss for the compactness of the cluster
    distance_loss : float, default = 1
        Loss for the distance between cluster centers for known classes
    center_step : float, default = 1
        How far the cluster center should move towards the batch cluster center

    Methods
    -------
    init_clusters(classes)
        Initialises cluster centers based on provided class values
    epoch() -> integer
        Updates network epoch
    """
    def __init__(
            self, save_num: int,
            states_dir: str,
            classes: Tensor,
            network: Network,
            unknown: int = 1,
            description: str = ''):
        """
        Parameters
        ----------
        save_num : integer
            File number to save the network
        states_dir : string
            Directory to save the network
        classes : C Tensor
            Classes of size C for clustering
        network : Network
            Network to predict low-dimensional data
        unknown : integer, default = 1
            Number of unknown classes
        description : string, default = ''
            Description of the network training
        """
        super().__init__(save_num, states_dir, network, description=description)
        self._unknown = unknown
        self._output = []
        self._classes = None
        self._cluster_centers = None
        self.classification = True
        self.sim_loss = 1
        self.compact_loss = 1
        self.distance_loss = 1
        self.center_step = 1

        self.init_clusters(classes)

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
            for _, labels, images, meta in loader:
                labels = labels.to(self._device)
                images = images.to(self._device)

                epoch_loss += self._loss(images, labels, meta)

        return epoch_loss / len(loader)

    def _cluster_loss(self, output: Tensor, labels: Tensor) -> Tensor:
        """
        Calculates the center of each class in the batch and shifts the class center towards the
        batch center

        Calculates the loss for the compactness of each batch class

        Parameters
        ----------
        output : NxD Tensor
            Cluster latent space from the network for batch size N and latent dimension D
        labels : N Tensor
            Class label for each output in the batch of size N

        Returns
        -------
        Tensor
            Loss for the cluster compactness for each class in the batch
        """
        batch_centers = []
        loss = torch.tensor(0.).to(self._device)
        batch_classes = torch.unique(labels)

        # Calculate cluster centers for each class in the batch
        for batch_class in batch_classes:
            cluster_idx = self._classes == batch_class
            label_idxs = labels == batch_class
            batch_centers.append(torch.mean(output[label_idxs], dim=0))

            # Shift class cluster center towards class batch center
            if self.train_state and self.center_step != 0:
                self._cluster_centers[cluster_idx] += ((batch_centers[-1] -
                                                       self._cluster_centers[cluster_idx]) *
                                                       self.center_step)

            # Calculate cluster compactness loss for the class batch
            loss += self.compact_loss * torch.mean((output[label_idxs] - batch_centers[-1]) ** 2)

        return loss

    def _distance_loss(self) -> Tensor:
        """
        Calculates the loss for the difference in distance between the classes and class cluster
        centers for known classes

        Returns
        -------
        Tensor
            Loss for the difference in distance for each class
        """
        known_classes = self._classes[self._unknown:]

        if len(known_classes) > 1 and self.distance_loss != 0:
            target_distances = torch.cdist(known_classes[:, None], known_classes[:, None], p=2)
            distances = torch.cdist(
                self._cluster_centers[self._unknown:],
                self._cluster_centers[self._unknown:],
                p=2,
            )
            # scaled_distances = distances * target_distances[0, -1] / distances[0, -1]
            return self.distance_loss * nn.MSELoss()(distances, target_distances)

        return torch.tensor(0.).to(self._device)

    def _loss(self, high_dim: Tensor, low_dim: Tensor, meta: Tensor) -> float:
        """
        Calculates the loss from the clusters

        Parameters
        ----------
        high_dim : Nx... Tensor
            Input high dimensional data of batch size N and the remaining dimensions depend on the
            network used
        low_dim : Nx1 Tensor
            Class label for each output in the batch of size N
        meta : NxM Tensor
            Extra information of size M for each output in the batch of size N

        Returns
        -------
        float
            Loss for the batch
        """
        loss = torch.tensor(0.).to(self._device)
        output = self.network(high_dim)

        # Save labels, metadata, and outputs for validation data
        if not self.train_state:
            self._output.extend(torch.cat((
                low_dim.cpu(),
                meta,
                output.detach().cpu(),
            ), dim=1))

        # Similarity loss between distances in feature space and distances in latent space
        if self.sim_loss != 0:
            loss += self.sim_loss * nn.MSELoss()(
                torch.cdist(output, output, p=2),
                torch.cdist(self.network.checkpoints[-2], self.network.checkpoints[-2], p=2),
            )

        # Shift class centers and calculate compactness loss
        loss += self._cluster_loss(output, low_dim.squeeze())

        # Loss for the difference in class values and cluster distances
        loss += self._distance_loss()

        # Update network
        if self.train_state:
            self.network.optimiser.zero_grad()
            loss.backward()
            self.network.optimiser.step()

        # Prevent memory leak from saving graphs generated by each batch
        self._cluster_centers = self._cluster_centers.detach()

        return loss.item()

    def init_clusters(self, classes: Tensor):
        """
        Initialises cluster centers based on provided class values

        Parameters
        ----------
        classes : C Tensor
            Class values of size C for which cluster centers will be based off
        """
        self._classes = classes.to(self._device)

        # Calculate the cluster centers for known classes, shift to origin, and set unknown
        # cluster centers to origin
        self._cluster_centers = torch.ones((
            len(self._classes),
            self.network.shapes[-1][-1],
        )).to(self._device) / torch.sqrt(torch.tensor(self.network.shapes[-1][-1]))
        self._cluster_centers *= (
                (self._classes - torch.mean(self._classes[self._unknown:]))[:, None]
        )
        self._cluster_centers[:self._unknown] *= 0

    def epoch(self) -> int:
        """
        Updates network epoch

        Returns
        -------
        integer
            Epoch number
        """
        self.network.epoch += 1
        self._output = []
        return self.network.epoch


class CompactClusterEncoder(BaseNetwork):
    """
    Semi supervised clustering of the latent space and prediction of labels

    Attributes
    ----------
    steps : integer
        Number of Markov chain steps for cluster identification
    classes : Tensor
        Classes available for clustering
    network : Network
        Encoder network
    train_state : boolean, default = True
        If network should be in the train or eval state
    cluster_loss : float, default = 0.1
        Weighting of cluster loss relative to label classification loss
    description : string, default = ''
        Description of the network training
    transform : tuple[float, float], default = None
        Expected label transformation of the encoder
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current encoder training and validation losses
    """
    def __init__(
            self,
            save_num: int,
            steps: int,
            states_dir: str,
            classes: Tensor,
            network: Network,
            description: str = ''):
        """
        Parameters
        ----------
        save_num : integer
            File number to save the network
        steps : integer
            Number of Markov chain steps for cluster identification
        states_dir : string
            Directory to save the network
        classes : Tensor
            Classes available for clustering
        network : Network
            Network to cluster predict low-dimensional data
        description : string, default = ''
            Description of the network training
        """
        super().__init__(save_num, states_dir, network, description=description)
        self.steps = steps
        self.classes = classes.to(self._device)
        self.cluster_loss = 1

    def _train_val(self, loaders: tuple[DataLoader, DataLoader]) -> float:
        epoch_loss = 0

        with torch.set_grad_enabled(self.train_state):
            for (images_l, labels), (images_u, _), *_ in zip(*loaders):
                labels = labels.to(self._device)
                images = torch.cat((images_l, images_u), dim=0).to(self._device)

                epoch_loss += self._loss(images, labels)

        return epoch_loss / (len(loaders[0]) + len(loaders[1]))

    def _label_propagation_cluster_loss(
            self,
            latent: Tensor,
            labels: Tensor) -> Tensor:
        """
        Calculates the label propagation for unlabelled data points and calculates the loss for the
        clusters

        Parameters
        ----------
        latent : Tensor
            Latent space to cluster, ordered by data points with known labels first
        labels : Tensor
            Labels for datapoints where labels are known

        Returns
        -------
        Tensor
            Cluster loss
        """
        loss = torch.tensor(0.).to(self._device)

        # Adjacency and transition matrices, softmax used instead of exponential
        # and row normalisation for stability
        adjacency = torch.matmul(latent, torch.t(latent))
        transition = nn.Softmax(dim=-1)(adjacency)
        transition = 1e-5 / transition.size(0) + (1 - 1e-5) * transition

        # Posterior calculation for label propagation
        if labels.size(0) < transition.size(0):
            transition_uu = transition[labels.size(0):, labels.size(0):]
            transition_ul = transition[labels.size(0):, :labels.size(0)]
            posterior_u = torch.matmul(torch.matmul(
                torch.linalg.inv(torch.eye(transition_uu.size(0)).to(self._device) - transition_uu),
                transition_ul,
            ), labels)
            posterior = torch.cat((labels, posterior_u))
        # elif labels.size(0):
        else:
            posterior = labels

        # Ideal compact clusters
        optimal_transition = torch.matmul(
            posterior,
            torch.t(posterior / torch.sum(posterior, dim=0)),
        )
        agreement = torch.matmul(posterior, torch.t(posterior))
        masked_transition = transition * agreement
        step_transition = transition

        # Markov chain for losses at different length graph chains
        for step in range(self.steps):
            if step:
                step_transition = torch.matmul(masked_transition, step_transition)

            loss -= torch.mean(
                optimal_transition * torch.log(step_transition),
            ) / self.steps

        return self.cluster_loss * loss

    def _loss(self, high_dim: Tensor, low_dim: Tensor) -> float:
        """
        Calculates the loss from the network's predictions and clusters

        Parameters
        ----------
        high_dim : Tensor
            Input high dimensional data
        low_dim : Tensor
            Labelled and unlabelled low dimensional data

        Returns
        -------
        float
            Loss for the network's predictions
        """
        # l_idxs = torch.argwhere(low_dim.squeeze() != -1).squeeze()
        # u_idxs = torch.argwhere(low_dim.squeeze() == -1).squeeze()
        # idxs = torch.cat((l_idxs, u_idxs))
        #
        # # Sort data and obtain network predictions
        # high_dim = high_dim[idxs]
        # low_dim = low_dim[idxs]

        # Obtain network predictions
        output = self.network(high_dim)
        latent = self.network.clone
        bottleneck = torch.argwhere(
            torch.count_nonzero(latent == 0, dim=0) == latent.size(0),
        ).flatten()

        # Remove part of latent space zeroed out due to information-ordered bottleneck layer
        if len(bottleneck) != 0:
            latent = latent[:, :bottleneck[0]]

        # Classification loss
        classes = nn.functional.one_hot(
            # torch.bucketize(low_dim, self.classes) - 1,
            # torch.bucketize(l_idxs.size(0), self.classes) - 1,
            low_dim,
            len(self.classes),
        ).squeeze().float()
        loss = nn.CrossEntropyLoss()(output[:low_dim.size(0)], classes)
        # _loss = nn.CrossEntropyLoss()(output[:l_idxs.size(0)], classes)

        # Cluster loss
        loss += self._label_propagation_cluster_loss(latent, classes)

        # Update network
        if self.train_state:
            self.network.optimiser.zero_grad()
            loss.backward()
            self.network.optimiser.step()

        return loss.item()
