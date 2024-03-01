"""
Classes that contain multiple types of networks
"""
import os
import pickle
import logging as log

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from netloader.network import Network
from zuko.flows import NSF
from numpy import ndarray

from src.networks.base import BaseNetwork
from src.utils.utils import get_device, save_name


class NormFlow(BaseNetwork):
    """
    Transforms a simple distribution into a distribution that reflects the input data

    Attributes
    ----------
    net : Network
        Neural spline flow
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
    batch_predict(data, samples[1e3]) -> Tensor
        Generates probability distributions for the data batch
    """
    def _loss(self, high_dim: Tensor, low_dim: Tensor) -> float:
        loss = -self.net().log_prob(high_dim).mean()
        self._update(loss)
        return loss.item()

    def batch_predict(
            self,
            data: Tensor,
            path: str = None,
            samples: list[int] = None,
            **__) -> ndarray:
        """
        Generates probability distributions for the data batch

        Parameters
        ----------
        data : Tensor
            Target distribution
        path : string, default = None
            Path as a pickle file to save the predictions if they should be saved
        samples : list[integer], default = [1e3]
            Number of samples to generate

        Returns
        -------
        ndarray
            Probability distributions for the given data
        """
        if samples is None:
            samples = [int(1e3)]

        # Generate samples
        distribution = self.net().sample(samples).squeeze(-1).cpu().numpy()

        if path:
            data = {'target': data, 'prediction': distribution}

            with open(path, 'wb') as file:
                pickle.dump(data, file)

        return distribution


class NormFlowEncoder(BaseNetwork):
    """
    Calculates the loss for a network and normalising flow that takes high-dimensional data
    and predicts a low-dimensional data distribution

    Attributes
    ----------
    flow : NSF
        Normalising flow to predict low-dimensional data distribution
    net : Network
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
    scheduler()
        Updates the scheduler for the flow and/or network if they are being trained
    predict(loader, path=None, samples=[1e3]) -> tuple[ndarray, ndarray, ndarray]
        Generates probability distributions for a dataset and can save to a file
    batch_predict(data, samples=[1e3]) -> Tensor
        Generates probability distributions for the given data batch
    """
    def __init__(
            self,
            save_num: int,
            states_dir: str,
            flow: NSF,
            net: Network,
            net_layers: int = None,
            description: str = ''):
        """
        Parameters
        ----------
        save_num : integer
            File number to save the flow
        states_dir : string
            Directory to save the network and flow
        flow : NSF
            Normalising flow to predict low-dimensional data distribution
        net : Network
            Network to condition the normalising flow from high-dimensional data
        net_layers : integer, default = None
            Number of layers in the network to use, if None use all layers
        description : string, default = ''
            Description of the network training
        """
        super().__init__(save_num, states_dir, net, description=description)
        self._net_layers = net_layers
        self.train_net = False
        self.train_flow = True
        self.flow = flow
        self.net = net
        self.flow.epoch = 0

        if save_num:
            self.save_path = save_name(save_num, states_dir, self.flow.name)

            if os.path.exists(self.save_path):
                log.warning(f'{self.save_path} already exists and will be overwritten '
                            f'if training continues')
        else:
            self.save_path = None

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
        self.net.layer_num = self._net_layers

        # Normalising flow loss
        output = self.net(high_dim)
        loss = -self.flow(output).log_prob(low_dim).mean()

        if not self.train_state:
            self.net.layer_num = None
            return loss.item()

        # Update network
        self.flow.optimiser.zero_grad()
        self.net.optimiser.zero_grad()
        loss.backward()

        if self.train_net:
            self.net.optimiser.step()

        if self.train_flow:
            self.flow.optimiser.step()

        # Remove network truncation
        self.net.layer_num = None

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
            self.net.epoch += 1

        if self.train_flow:
            return self.flow.epoch

        return self.net.epoch

    def scheduler(self):
        """
        Updates the scheduler for the flow and/or network if they are being trained
        """
        if self.train_net:
            super().scheduler()

        if self.train_flow:
            self.flow.scheduler.step(self.losses[1][-1])

    def predict(
            self,
            loader: DataLoader,
            path: str = None,
            samples: list[int] = None,
            **_) -> tuple[ndarray, ndarray, ndarray]:
        """
        Generates probability distributions for a dataset and can save to a file

        Parameters
        ----------
        loader : DataLoader
            Dataset to generate predictions for
        path : string, default = None
            Path as a CSV file to save the predictions if they should be saved
        samples : list[integer], default = [1e3]
            Number of samples to generate from the predicted distribution

        Returns
        -------
        tuple[N ndarray, Nx1 ndarray, NxS ndarray]
            Prediction IDs, target values and predicted distribution with samples S for dataset of
            size N
        """
        bin_num = 100
        header = 'IDs,Targets,Probabilities,Maxima,Medians,Distributions'
        probs = []
        maxima = []
        ids, targets, distributions = super().predict(loader, samples=samples)

        if not path:
            return ids, targets, distributions

        medians = np.median(distributions, axis=-1)

        for target, distribution in zip(targets, distributions):
            hist, bins = np.histogram(distribution, bins=bin_num, density=True)
            prob = hist * (bins[1] - bins[0])
            bins[-1] += 1e-6
            probs.append(prob[np.clip(np.digitize(target, bins) - 1, 0, bin_num - 1)])
            maxima.append(bins[np.argmax(hist)])

        output = np.hstack((
            ids[:, np.newaxis],
            targets,
            np.array(probs),
            np.expand_dims(maxima, axis=1),
            medians[:, np.newaxis],
            distributions,
        ))
        np.savetxt(path, output, delimiter=',', fmt='%s', header=header)

        return ids, targets, distributions

    def batch_predict(self, data: Tensor, samples: list[int] = None, **_) -> Tensor:
        """
        Generates probability distributions for the data batch

        Parameters
        ----------
        data : Tensor
            Data to generate distributions for
        samples : list[integer], default = [1e3]
            Number of samples to generate

        Returns
        -------
        Tensor
            Probability distributions for the given data
        """
        if samples is None:
            samples = [1e3]

        # Temporarily truncate the network and generate samples
        self.net.layer_num = self._net_layers
        samples = torch.transpose(
            self.flow(self.net(data)).sample([samples]).squeeze(-1),
            0,
            1,
        )

        # Remove network truncation
        self.net.layer_num = None
        return samples


def norm_flow(
        features: int,
        transforms: int,
        learning_rate: float,
        hidden_features: list[int],
        context: int = 0) -> NSF:
    """
    Generates a neural spline flow (NSF) for use in BaseNetwork

    Adds attributes of name ('flow'), optimiser (Adam), and scheduler (ReduceLROnPlateau)

    Parameters
    ----------
    features : integer
        Dimensions of the probability distribution
    transforms : integer
        Number of transforms
    learning_rate : float
        Learning rate of the NSF
    hidden_features : list[integer]
        Number of features in each of the hidden layers
    context : integer, default = 0
        Number of features to condition the NSF

    Returns
    -------
    NSF
        Neural spline flow with attributes required for training
    """
    device = get_device()[1]
    flow = NSF(
        features=features,
        context=context,
        transforms=transforms,
        hidden_features=hidden_features,
    ).to(device)

    flow.name = 'flow'
    flow._device = device
    flow.optimiser = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    flow.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        flow.optimiser,
        patience=5,
        factor=0.5,
        verbose=True,
    )
    return flow
