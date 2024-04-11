"""
Architectures that cluster data
"""
import torch
import numpy as np
from numpy import ndarray
from torch import Tensor, nn
from torch.utils.data import DataLoader
from netloader.network import Network

from src.networks.base import BaseNetwork
from src.utils.utils import label_change


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
    classes : (C) Tensor
        Classes of size C for clustering
    net : Network
        Neural network
    train_state : boolean, default = True
        If network should be in the train or eval state
    classify : boolean, default = True
        If the network should be trained to classify
    cluster : boolean, default = False
        If the network should be trained to cluster
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
    description : string, default = ''
        Description of the network training
    transform : tuple[float, float], default = None
        Expected low-dimensional data transformation of the network
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current network training and validation losses

    Methods
    -------
    init_clusters(classes)
        Initialises cluster centers based on provided class values
    predict(loader, path=None) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]
        Generates predictions and latent space for a dataset and can save to a file
    batch_predict(data) -> tuple[Tensor, Tensor, Tensor]
        Generates predictions and latent space for the given data batch
    """
    def __init__(
            self,
            save_num: int,
            states_dir: str,
            classes: Tensor,
            net: Network,
            unknown: int = 1,
            method: str = 'mean',
            description: str = ''):
        """
        Parameters
        ----------
        save_num : integer
            File number to save the network
        states_dir : string
            Directory to save the network
        classes : (C) Tensor
            Classes of size C for clustering
        net : Network
            Network to predict low-dimensional data
        unknown : integer, default = 1
            Number of unknown classes
        method : string, default = 'mean'
            Whether to calculate the center of a cluster using the 'mean' or 'median'
        description : string, default = ''
            Description of the network training
        """
        super().__init__(save_num, states_dir, net, description=description)
        self._unknown = unknown
        self._method = method
        self._output = []
        self._cluster_centers = None
        self.classify = True
        self.cluster = True
        self.sim_loss = 0.7
        self.compact_loss = 0.5
        self.distance_loss = 3
        self.class_loss = 0.2
        self.classes = classes.to(self._device)
        self.scale = nn.Parameter(torch.tensor((1.,), requires_grad=True, device=self._device))
        self.center_step = torch.ones(len(self.classes), device=self._device)

        self.net.optimiser.param_groups[0]['params'].append(self.scale)

    def _init_clusters(self, latent_dim: int):
        """
        Initialises cluster centers based on provided class values

        Parameters
        ----------
        latent_dim : integer
            Dimension of the latent space
        """
        self._cluster_centers = torch.zeros((len(self.classes), latent_dim)).to(self._device)
        self._cluster_centers[:, 0] = self.classes
        self._cluster_centers[:self._unknown, 0] = 0.5

    def _cluster_centers_loss(self, latent: Tensor, labels: Tensor):
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
        loss = torch.tensor(0.).to(self._device)

        # Calculate cluster centers for each class in the batch
        for batch_class in torch.unique(labels)[self._unknown:]:
            # Shift class cluster center towards class batch center
            if self.train_state and torch.any(self.center_step):
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
        class_idx = self.classes == cluster_class
        class_vecs = latent[labels == cluster_class] - self._cluster_centers[class_idx]

        # Standard deviation in the direction to the closest class
        direc = torch.min(
            torch.abs(
                self._cluster_centers[~class_idx][self._unknown:] -
                self._cluster_centers[class_idx],
            ),
            dim=0,
        )[0]

        return self.compact_loss * torch.std(torch.linalg.norm(torch.multiply(
            (torch.sum(class_vecs * direc, dim=1) / torch.dot(direc, direc))[:, None],
            direc[None],
        ), dim=1))

    def _distance_loss(self) -> Tensor:
        """
        Calculates the loss for the difference in distance between the classes and class cluster
        centers for known classes

        Returns
        -------
        Tensor
            Loss for the difference in distance for each class
        """
        known_classes = self.classes[self._unknown:]

        if len(known_classes) > 1 and self.distance_loss:
            centers = self.scale * self._cluster_centers[self._unknown:, 0]
            return self.distance_loss * nn.MSELoss()(centers, known_classes)

        return torch.tensor(0., device=self._device)

    def _update_centers(self, batch_class: Tensor, latent: Tensor, labels: Tensor):
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
        class_idx = self.classes == batch_class
        label_idxs = labels == batch_class

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
        loss = torch.tensor(0.).to(self._device)
        output = self.net(in_data)
        latent = self.net.checkpoints[-1]

        if self._cluster_centers is None:
            self._init_clusters(latent.size(1))

        # Similarity loss between distances in feature space and distances in latent space
        if (self.cluster and self.sim_loss and
                torch.isin(target, self.classes[:self._unknown]).any()):
            idxs = target.flatten() == self.classes[:self._unknown]
            loss += self.sim_loss * nn.MSELoss()(
                torch.cdist(latent[idxs, :1], latent[~idxs, :1]),
                torch.cdist(self.net.checkpoints[-3][idxs], self.net.checkpoints[-3][~idxs]),
            )

        if self.cluster:
            # Shift class centers and calculate compactness loss
            loss += self._cluster_centers_loss(latent, target.squeeze())

            # Loss for the difference in class values and cluster distances
            loss += self._distance_loss()

        # Classification loss
        if self.classify and self.class_loss:
            one_hot = label_change(target.squeeze(), self.classes)
            loss += self.class_loss * nn.CrossEntropyLoss()(output, one_hot)

        # Update network
        self._update(loss)

        # Prevent memory leak from saving graphs generated by each batch
        self._cluster_centers = self._cluster_centers.detach()

        return loss.item()

    def predict(
            self,
            loader: DataLoader,
            path: str = None,
            header: list[str] = None,
            **_) -> dict:
        """
        Generates predictions and latent space for a dataset and can save to a file

        Parameters
        ----------
        loader : DataLoader
            Dataset to generate predictions and latent space for
        path : string, default = None
            Path as a CSV file to save the predictions if they should be saved
        header : list[string], default = ['ids', 'targets', 'preds', 'probs', 'latent']
            Header for the predicted data, only used by child classes

        Returns
        -------
        list[ndarray]
            Prediction IDs, target classes, predicted classes, class probabilities for C classes,
            and latent space of dimension D for dataset of size N
        """
        if header is None:
            header = ['ids', 'targets', 'preds', 'probs', 'latent']

        # Transform values
        data = super().predict(loader, header=header)
        data['latent'][:, 0] = (data['latent'][:, 0] * self.scale.detach().cpu().numpy() *
                                self.transform[1] + self.transform[0])
        accuracy = np.count_nonzero(data['targets'].flatten() == data['preds']) / len(data['ids'])
        print(f"Accuracy: {accuracy:.1%}")
        self._save_predictions(path, data)
        return data

    def batch_predict(self, data: Tensor, **_) -> tuple[ndarray, ndarray, ndarray]:
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
        probs = self.net(data)
        predicts = label_change(
            torch.argmax(probs, dim=1),
            torch.arange(probs.size(1)).to(self._device),
            out_label=self.classes,
        )
        return (
            predicts.detach().cpu().numpy(),
            probs.detach().cpu().numpy(),
            self.net.checkpoints[-1].detach().cpu().numpy(),
        )


class CompactClusterEncoder(ClusterEncoder):
    """
    Semi supervised clustering of the latent space and prediction of labels

    Attributes
    ----------
    classes : (C) Tensor
        Classes of size C for clustering
    net : Network
        Encoder network
    train_state : boolean, default = True
        If network should be in the train or eval state
    cluster_loss : float, default = 1
        Weighting of the cluster loss
    distance_loss : float, default = 1
        Loss weight for the distance between cluster centers for known classes
    class_loss : float, default = 1
        Weighting of the cross entropy loss
    center_step : float, default = 1
        How far the cluster center should move towards the batch cluster center, if 0, cluster
        centers will be fixed
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
            states_dir: str,
            classes: Tensor,
            net: Network,
            steps: int = 3,
            unknown: int = 1,
            method: str = 'mean',
            description: str = ''):
        """
        Parameters
        ----------
        save_num : integer
            File number to save the network
        states_dir : string
            Directory to save the network
        classes : C Tensor
            Classes available for clustering for C possible classes
        net : Network
            Network to cluster predict low-dimensional data
        steps : integer, default = 3
            Number of Markov chain steps for cluster identification
        unknown : integer, default = 1
            Number of unknown classes
        method : string, default = 'mean'
            Whether to calculate the center of a cluster using the 'mean' or 'median'
        description : string, default = ''
            Description of the network training
        """
        super().__init__(
            save_num,
            states_dir,
            classes,
            net,
            unknown=unknown,
            method=method,
            description=description,
        )
        self._steps = steps
        self.sim_loss = 0  # Unused
        self.compact_loss = 0  # Unused
        self.class_loss = 0.2
        self.distance_loss = 2
        self.cluster_loss = 2.2

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
        loss = torch.tensor(0.).to(self._device)

        # Adjacency and transition matrices, softmax used instead of exponential
        # and row normalisation for stability
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

        return self.cluster_loss * loss

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
        batch_centers = []
        known_classes = self.classes[self._unknown:]
        batch_classes = known_classes[torch.isin(known_classes, torch.unique(labels))]

        if len(known_classes) == 0:
            return torch.tensor(0.).to(self._device)

        # Calculate cluster centers for each class in the batch
        for batch_class in batch_classes:
            label_idxs = labels == batch_class

            if self._method == 'mean':
                batch_centers.append(torch.mean(latent[label_idxs, 0]))
            else:
                batch_centers.append(torch.median(latent[label_idxs, 0]))

        return self.distance_loss * nn.MSELoss()(
            self.scale * torch.stack(batch_centers),
            batch_classes,
        )

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
        target = target.squeeze()
        l_idxs = torch.isin(target, self.classes)

        # Obtain network predictions
        output = self.net(in_data)
        latent = self.net.checkpoints[-1]
        bottleneck = torch.argwhere(
            torch.count_nonzero(latent == 0, dim=0) == latent.size(0),
        ).flatten()

        if self._cluster_centers is None:
            self._init_clusters(latent.size(1))

        # Remove part of latent space zeroed out due to information-ordered bottleneck layer
        if len(bottleneck) != 0:
            latent = latent[:, :bottleneck[0]]

        # Classification loss
        one_hot = label_change(target[l_idxs], self.classes, one_hot=True).float()
        loss = self.class_loss * nn.CrossEntropyLoss()(output[l_idxs], one_hot)

        # Cluster loss
        loss += self._label_propagation_cluster_loss(latent[l_idxs], one_hot)

        # Distance loss
        if self.distance_loss:
            loss += self._cluster_centers_loss(latent[l_idxs], target[l_idxs])

        self._update(loss)

        # Prevent memory leak from saving graphs generated by each batch
        self._cluster_centers = self._cluster_centers.detach()

        return loss.item()
