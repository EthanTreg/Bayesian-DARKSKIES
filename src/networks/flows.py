"""
Classes that contain multiple types of networks
"""
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from netloader.network import Network
from zuko.flows import NSF
from numpy import ndarray

from src.networks.base import BaseNetwork
from src.utils.utils import get_device


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
    def _data_loader_translation(self, low_dim: Tensor, high_dim: Tensor) -> tuple[Tensor, Tensor]:
        """
        Takes the low and high dimensional tensors from the data loader, and orders them as inputs
        and targets for the network

        Parameters
        ----------
        low_dim : Tensor
            Low dimensional tensor from the data loader
        high_dim : Tensor
            High dimensional tensor from the data loader

        Returns
        -------
        tuple[Tensor, Tensor]
            Input and output target tensors
        """
        return low_dim, high_dim

    def _loss(self, _: Tensor, target: Tensor) -> float:
        """
        Calculates the loss from the flow's predictions

        Parameters
        ----------
        target : (N, ...) Tensor
            Target high dimensional data of batch size N and the remaining dimensions depend on the
            network used

        Returns
        -------
        float
            Loss from the network's predictions
        """
        loss = -self.net().log_prob(target).mean()
        self._update(loss)
        return loss.item()

    def predict(
            self,
            path: str = None,
            samples: list[int] = None,
            header: list[str] = None,
            **_) -> dict:
        """

        Parameters
        ----------
        path : string, default = None
            Path as CSV file to save the predictions if they should be saved
        samples : list[integer], default = [1e3]
            Number of samples to generate
        header : list[string], default = ['samples']
            Header for the predicted data, only used by child classes

        Returns
        -------
        dictionary
            Predicted distribution
        """
        if header is None:
            header = ['samples']

        if samples is None:
            samples = [int(1e3)]

        data = {header[0]: self.net().sample(samples).moveaxis(0, -1).detach().cpu().numpy()}
        self._save_predictions(path, data)
        return data


class NormFlowEncoder(BaseNetwork):
    """
    Calculates the loss for a network and normalising flow that takes high-dimensional data
    and predicts a low-dimensional data distribution

    Attributes
    ----------
    net : NSF
        Normalising flow to predict low-dimensional data distribution
    net : Network
        Network to condition the normalising flow from high-dimensional data
    train_state : boolean, default = True
        If network should be in the train or eval state
    train_flow : boolean, default = True
        If normalising flow should be trained
    train_encoder : boolean, default = False
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
            net: Network,
            encoder: Network,
            net_layers: int = None,
            description: str = ''):
        """
        Parameters
        ----------
        save_num : integer
            File number to save the flow
        states_dir : string
            Directory to save the network and flow
        net : Network
            Normalising flow to predict low-dimensional data distribution
        encoder : Network
            Network to condition the normalising flow from high-dimensional data
        net_layers : integer, default = None
            Number of layers in the network to use, if None use all layers
        description : string, default = ''
            Description of the network training
        """
        super().__init__(save_num, states_dir, net, description=description)
        self._net_layers = net_layers
        self.train_encoder = False
        self.train_flow = True
        self.encoder = encoder
        self.encoder.epoch = 0

    def _loss(self, in_data: Tensor, target: Tensor) -> float:
        """
        Calculates the loss from the network and flow's predictions

        Parameters
        ----------
        in_data : (N,...) Tensor
            Input high dimensional data of batch size N and the remaining dimensions depend on the
            network used
        target : (N, ...) Tensor
            Target low dimensional data of batch size N and the remaining dimensions depend on the
            network used

        Returns
        -------
        float
            Loss from the flow's predictions'
        """
        # Temporarily truncate the network
        self.encoder.layer_num = self._net_layers

        # Normalising flow loss
        loss = -self.net(self.encoder(in_data)).log_prob(target).mean()

        if not self.train_state:
            self.encoder.layer_num = None
            return loss.item()

        # Update network
        self.net.optimiser.zero_grad()
        self.encoder.optimiser.zero_grad()
        loss.backward()

        if self.train_encoder:
            self.encoder.optimiser.step()

        if self.train_flow:
            self.net.optimiser.step()

        # Remove network truncation
        self.encoder.layer_num = None

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
            self.encoder.train()
        else:
            self.encoder.eval()

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
        super().load(load_num[1], states_dir, network=self.encoder)

    def epoch(self) -> int:
        """
        Updates network and flow epoch if they are being trained

        Returns
        -------
        integer
            Epoch number
        """
        if self.train_flow:
            self.net.epoch += 1

        if self.train_encoder:
            self.encoder.epoch += 1

        if self.train_flow:
            return self.net.epoch

        return self.encoder.epoch

    def scheduler(self):
        """
        Updates the scheduler for the flow and/or network if they are being trained
        """
        if self.train_flow:
            super().scheduler()

        if self.train_encoder:
            self.encoder.scheduler.step(self.losses[1][-1])

    def predict(
            self,
            loader: DataLoader,
            bin_num: int = 100,
            path: str = None,
            samples: list[int] = None,
            header: list[str] = None,
            **_) -> dict:
        """
        Generates probability distributions for a dataset and can save to a file

        Parameters
        ----------
        loader : DataLoader
            Dataset to generate predictions for
        bin_num : integer, default = 100
            Number of bins for calculating the probability of the target and maximum of the
            distribution, higher is more precise but requires more samples
        path : string, default = None
            Path as a pkl file to save the predictions if they should be saved
        samples : list[integer], default = [1e3]
            Number of samples to generate from the predicted distribution
        header : list[string], default = ['ids', 'targets', 'probs', 'max', 'meds', 'distributions']
            Header for the predicted data, only used by child classes

        Returns
        -------
        dictionary
            Prediction IDs, target values, target probability, distribution maximum, distribution
            median, and predicted distribution with samples S for dataset of size N
        """
        probs = []
        maxima = []
        data = super().predict(loader, samples=samples)

        if header is None:
            header = ['ids', 'targets', 'probs', 'max', 'meds', 'distributions']

        for target, distribution in zip(data['targets'], data['distributions']):
            hist, bins = np.histogram(distribution, bins=bin_num, density=True)
            prob = hist * (bins[1] - bins[0])
            bins[-1] += 1e-6
            probs.append(prob[np.clip(np.digitize(target, bins) - 1, 0, bin_num - 1)])
            maxima.append(bins[np.argmax(hist)])

        data[header[2]] = np.stack(probs)
        data[header[3]] = np.stack(maxima)
        data[header[4]] = np.median(data[-1], axis=-1)
        self._save_predictions(path, data)
        return data

    def batch_predict(self, data: Tensor, samples: list[int] = None, **_) -> tuple[ndarray]:
        """
        Generates probability distributions for the data batch

        Parameters
        ----------
        data : (N,...) Tensor
            Data to generate distributions for
        samples : list[integer], default = [1e3]
            Number of samples to generate

        Returns
        -------
        tuple[(N,S) ndarray]
            S samples from N probability distributions for the given data
        """
        if samples is None:
            samples = [1e3]

        # Temporarily truncate the network and generate samples
        self.encoder.layer_num = self._net_layers
        samples = torch.transpose(
            self.net(self.encoder(data)).sample([samples]).squeeze(-1),
            0,
            1,
        )

        # Remove network truncation
        self.encoder.layer_num = None
        return (samples.detach().cpu().numpy(),)


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

    flow._device = device
    flow.name = 'flow'
    flow.optimiser = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    flow.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        flow.optimiser,
        patience=5,
        factor=0.5,
        verbose=True,
    )
    return flow
