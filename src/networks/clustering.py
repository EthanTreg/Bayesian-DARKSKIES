"""
Architectures that cluster data
"""
from time import time

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
    classify : boolean, default = True
        If the network should be trained to classify
    cluster : boolean, default = False
        If the network should be trained to cluster
    sim_loss : float, default = 1
        Loss for the difference between the distance in feature space and latent space
    compact_loss : float, default = 1
        Loss for the compactness of the cluster
    distance_loss : float, default = 1
        Loss for the distance between cluster centers for known classes
    classify_loss : boolean, default = True
        Loss for the class classification
    center_step : float, default = 1
        How far the cluster center should move towards the batch cluster center, if 0, cluster
        centers will be fixed

    Methods
    -------
    init_clusters(classes)
        Initialises cluster centers based on provided class values
    epoch() -> integer
        Updates network epoch
    """
    def __init__(
            self,
            save_num: int,
            latent_dim: int,
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
        latent_dim : integer
            Dimension of the latent space
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
        self.classify = True
        self.cluster = False
        self.sim_loss = 0
        self.compact_loss = 1
        self.distance_loss = 0
        self.classify_loss = 1
        self.center_step = torch.ones(len(classes)).to(self._device)

        self.init_clusters(latent_dim, classes)

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
            if self.train_state and torch.any(self.center_step):
                self._cluster_centers[cluster_idx] += ((batch_centers[-1] -
                                                       self._cluster_centers[cluster_idx]) *
                                                       self.center_step[cluster_idx])

            # Calculate cluster compactness loss for the class batch
            # loss += self.compact_loss * torch.mean((output[label_idxs] - batch_centers[-1]) ** 2)
            loss += self.compact_loss * torch.mean(
                (output[label_idxs] - self._cluster_centers[cluster_idx]) ** 2
            )

        return loss / len(batch_classes)

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

        if len(known_classes) > 1 and self.distance_loss:
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
        latent = self.network.checkpoints[-1]

        # Save labels, metadata, and outputs for validation data
        if not self.train_state:
            self._output.extend(torch.cat((
                low_dim.cpu(),
                meta,
                latent.detach().cpu(),
            ), dim=1))

        # Similarity loss between distances in feature space and distances in latent space
        if self.cluster and self.sim_loss:
            loss += self.sim_loss * nn.MSELoss()(
                torch.cdist(latent, latent, p=2),
                torch.cdist(self.network.checkpoints[-2], self.network.checkpoints[-2], p=2),
            )

        if self.cluster:
            # Shift class centers and calculate compactness loss
            loss += self._cluster_loss(latent, low_dim.squeeze())

            # Loss for the difference in class values and cluster distances
            loss += self._distance_loss()

        # Classification loss
        if self.classify and self.classify_loss:
            one_hot = label_change(low_dim.squeeze(), self._classes)
            loss += self.classify_loss * nn.CrossEntropyLoss()(output, one_hot)

        # Update network
        if self.train_state:
            self.network.optimiser.zero_grad()
            loss.backward()
            self.network.optimiser.step()

        # Prevent memory leak from saving graphs generated by each batch
        self._cluster_centers = self._cluster_centers.detach()

        return loss.item()

    def init_clusters(self, latent_dim: int, classes: Tensor):
        """
        Initialises cluster centers based on provided class values

        Parameters
        ----------
        latent_dim : integer
            Dimension of the latent space
        classes : C Tensor
            Class values of size C for which cluster centers will be based off
        """
        self._classes = classes.to(self._device)
        self._cluster_centers = torch.zeros((len(self._classes), latent_dim)).to(self._device)
        self._cluster_centers[:, 0] = self._classes
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
    network : Network
        Encoder network
    train_state : boolean, default = True
        If network should be in the train or eval state
    cluster_loss : float, default = 1
        Weighting of the cluster loss
    class_loss : float, default = 1
        Weighting of the cross entropy loss
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
            network: Network,
            steps: int = 3,
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
        network : Network
            Network to cluster predict low-dimensional data
        steps : integer, default = 3
            Number of Markov chain steps for cluster identification
        description : string, default = ''
            Description of the network training
        """
        super().__init__(save_num, states_dir, network, description=description)
        self._steps = steps
        self._classes = classes.to(self._device)
        self.cluster_loss = 2.2
        self.class_loss = 0.2

    def _label_propagation_cluster_loss(self, latent: Tensor, one_hot: Tensor) -> Tensor:
        """
        Calculates the label propagation for unlabelled data points and calculates the loss for the
        clusters

        Parameters
        ----------
        latent : NxD Tensor
            Latent space to cluster, ordered by data points with known labels first
        one_hot : LxC Tensor
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
            torch.t(posterior / torch.sum(posterior, dim=0)),
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
        low_dim = low_dim.squeeze()
        l_idxs = torch.isin(low_dim, self._classes)

        # Obtain network predictions
        output = self.network(high_dim)
        latent = self.network.checkpoints[-1]
        latent = torch.cat((latent[l_idxs], latent[~l_idxs]))
        bottleneck = torch.argwhere(
            torch.count_nonzero(latent == 0, dim=0) == latent.size(0),
        ).flatten()

        # Remove part of latent space zeroed out due to information-ordered bottleneck layer
        if len(bottleneck) != 0:
            latent = latent[:, :bottleneck[0]]

        # Classification loss
        one_hot = label_change(low_dim[l_idxs], self._classes, one_hot=True).float()
        loss = self.class_loss * nn.CrossEntropyLoss()(output[l_idxs], one_hot)

        # Cluster loss
        loss += self._label_propagation_cluster_loss(latent, one_hot)
        self._update(loss)
        return loss.item()

    def predict(
            self,
            loader: DataLoader,
            path: str = None) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
        """
        Generates predictions and latent space for a dataset and can save to a file

        Parameters
        ----------
        loader : DataLoader
            Dataset to generate predictions and latent space for
        path : string, default = None
            Path to save the predictions if they should be saved

        Returns
        -------
        tuple[N ndarray, N ndarray, N ndarray, NxC ndarray, NxD ndarray]
            Prediction IDs, target classes, predicted classes, class probabilities for C classes,
            and latent space of dimension D for dataset of size N
        """
        initial_time = time()
        ids = []
        probs = []
        targets = []
        latents = []
        predicts = []
        self.train(False)

        # Generate predictions
        with torch.no_grad():
            for id_batch, target, images, *_ in loader:
                ids.extend(id_batch)
                targets.extend(target)
                predict, prob, latent = self.batch_predict(images.to(self._device))
                probs.extend(prob.cpu())
                latents.extend(latent.cpu())
                predicts.extend(predict.cpu())

        # Transform values
        ids = np.array(ids)
        probs = torch.stack(probs).numpy()
        targets = torch.stack(targets).numpy().squeeze() * self.transform[1] + self.transform[0]
        latents = torch.stack(latents).numpy()
        predicts = torch.stack(predicts).numpy() * self.transform[1] + self.transform[0]
        header = f"IDs,Targets,Predictions,Probabilities{',' * probs.shape[1]}Latent"
        print(f'Prediction time: {time() - initial_time:.3e} s\t'
              f'Accuracy: {np.count_nonzero(targets == predicts) / len(targets):.1%}')

        if path:
            output = np.hstack((
                ids[:, np.newaxis],
                targets[:, np.newaxis],
                predicts[:, np.newaxis],
                probs,
                latents,
            ))
            np.savetxt(path, output, delimiter=',', fmt='%s', header=header)

        return ids, targets, predicts, probs, latents

    def batch_predict(self, data: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Generates predictions and latent space for the given data batch

        Parameters
        ----------
        data : Nx... Tensor
            N data to generate predictions for

        Returns
        -------
        tuple[N Tensor, NxC Tensor, NxD Tensor]
            N predictions, prediction probabilities for C classes and latent space points of
            dimension D for the given data
        """
        probs = self.network(data)
        predicts = label_change(
            torch.argmax(probs, dim=1),
            torch.arange(probs.size(1)).to(self._device),
            out_label=self._classes,
        )
        return predicts, probs, self.network.checkpoints[-1]
