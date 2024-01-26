"""
Loads data and creates data loaders for network training
"""
import pickle

import torch
import numpy as np
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import v2

from src.utils.utils import get_device


class DarkDataset(Dataset):
    """
    A dataset object containing image maps and dark matter cross-sections for PyTorch training

    Attributes
    ----------
    ids : ndarray
        IDs for each cluster in the dataset
    classes : Tensor
        Supervised class boundaries for one hot classification
    labels : Tensor
        Supervised labels for dark matter cross-section for each cluster
    images : Tensor
        Lensing and X-ray maps for each cluster
    meta : Tensor
        Metadata for each cluster in the dataset
    aug : Compose
        Image augmentation transform
    transform : tuple[float, float], default = None
        Label min and range for 0-1 normalisation
    idxs : ndarray, default = None
        Data indices for random training & validation datasets

    Methods
    -------
    normalise()
        Normalises the labels
    """
    def __init__(self, data_path: str, sims: list[str]):
        """
        Parameters
        ----------
        data_path : string
            Path to the data file with the cluster dataset
        sims : list[str]
            Which simulations to load
        """
        idxs = []
        self.idxs = None
        self.transform = None

        # Load data from file
        with open(data_path, 'rb') as file:
            labels, images = pickle.load(file)

        # Get specified sim data
        for sim in sims:
            idxs.extend(np.argwhere(labels['sim'] == sim).flatten())

        # Remove stellar maps
        self.images = np.moveaxis(images[idxs], 3, 1)

        # Create labels
        self.labels = labels['label'][idxs]
        self.labels[self.labels == 0.3] = -1
        self.classes = torch.from_numpy(np.append(np.unique(self.labels), np.max(self.labels) * 2))

        # Metadata
        del labels['galaxy_catalogues']
        del labels['sim']
        self.meta = torch.from_numpy(np.array(
            list(labels.values()),
            dtype=float,
        )).swapaxes(0, 1)[idxs]

        # Uses cluster IDs if provided, otherwise, number dataset in order
        if 'clusterID' in labels:
            self.ids = labels['clusterID'][idxs]
        else:
            self.ids = np.arange(self.images.shape[0])

        self.labels = torch.from_numpy(self.labels).float()[:, None]
        self.images = torch.from_numpy(self.images).float()

        # Image augmentations
        self.aug = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=(0, 180)),
        ])

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[ndarray, Tensor, Tensor, Tensor]:
        """
        Gets the training data for the given index

        Parameters
        ----------
        idx : integer
            Index of the target cluster

        Returns
        -------
        tuple[ndarray, Tensor, Tensor, ndarray]
            Cluster ID, cluster label, augmented image map, and metadata
        """
        return self.ids[idx], self.labels[idx], self.aug(self.images[idx]), self.meta[idx]

    def normalise(self, idxs: list[int] = None, transform: tuple[float, float] = None):
        """
        Normalises the labels

        Parameters
        ----------
        idxs : list[integer]
            Which indices should be normalised
        transform : tuple[float, float], default = None
            Pre-defined transformation
        """
        if idxs is None:
            self.labels, self.transform = data_normalisation(
                self.labels,
                mean=False,
                transform=transform,
            )
        else:
            self.labels[idxs], self.transform = data_normalisation(
                self.labels[idxs],
                mean=False,
                transform=transform,
            )


def _ndarray_normalisation(
        data: ndarray,
        mean: bool = True,
        axis: int = None,
        transform: tuple[float, float] | tuple[ndarray, ndarray] = None,
) -> tuple[ndarray, tuple[float, float] | tuple[ndarray, ndarray]]:
    """
    Transforms ndarray data either by normalising or
    scaling between 0 & 1 depending on if mean is true or false.

    Parameters
    ----------
    data : ndarray
        Data to be normalised
    mean : boolean, default = True
        If data should be normalised or scaled between 0 and 1
    axis : integer, default = None
        Which axis to normalise over, if none, normalise over all axes
    transform: tuple[float, float] | tuple[ndarray, ndarray], default = None
        If transformation values exist already

    Returns
    -------
    tuple[ndarray, tuple[float, float] | tuple[ndarray, ndarray]]
        Transformed data & transform values
    """
    if len(np.unique(data)) == 1:
        return data, transform or (0, 1)

    if mean and not transform:
        transform = [np.mean(data, axis=axis), np.std(data, axis=axis)]
    elif not mean and not transform:
        transform = [
            np.min(data, axis=axis),
            np.max(data, axis=axis) - np.min(data, axis=axis)
        ]

    if axis:
        data = (data - np.expand_dims(transform[0], axis=axis)) /\
               np.expand_dims(transform[1], axis=axis)
    else:
        data = (data - transform[0]) / transform[1]

    return data, transform


def _tensor_normalisation(
        data: Tensor,
        mean: bool = True,
        dim: int = None,
        transform: tuple[float, float] | tuple[Tensor, Tensor] = None,
) -> tuple[Tensor, tuple[float, float] | tuple[Tensor, Tensor]]:
    """
    Transforms Tensor data either by normalising or
    scaling between 0 & 1 depending on if mean is true or false.

    Parameters
    ----------
    data : Tensor
        Data to be normalised
    mean : boolean, default = True
        If data should be normalised or scaled between 0 and 1
    dim : integer, default = None
        Which dimension to normalise over, if none, normalise over all dimensions
    transform: tuple[float, float] | tuple[Tensor, Tensor], default = None
        If transformation values exist already

    Returns
    -------
    tuple[Tensor, tuple[float, float] | tuple[Tensor, Tensor]]
        Transformed data & transform values
    """
    if len(torch.unique(data)) == 1:
        return data, transform or (0, 1)

    if mean and not transform:
        transform = [torch.mean(data, dim=dim), torch.std(data, dim=dim)]
    elif not mean and not transform and dim:
        transform = [
            torch.min(data, dim=dim)[0],
            torch.max(data, dim=dim)[0] - torch.min(data, dim=dim)[0]
        ]
    elif not mean and not transform:
        transform = [torch.min(data).item(), torch.max(data).item() - torch.min(data).item()]

    if dim:
        data = (data - transform[0].unsqueeze(dim)) / transform[1].unsqueeze(dim)
    else:
        data = (data - transform[0]) / transform[1]

    return data, transform


def data_normalisation(
        data: ndarray | Tensor,
        mean: bool = True,
        axis: int = None,
        transform: tuple[float, float] = None,
) -> tuple[ndarray | Tensor, tuple[float, float] | tuple[ndarray, ndarray] | tuple[Tensor, Tensor]]:
    """
    Transforms data either by normalising or
    scaling between 0 & 1 depending on if mean is true or false.

    Parameters
    ----------
    data : ndarray | Tensor
        Data to be normalised
    mean : boolean, default = True
        If data should be normalised or scaled between 0 and 1
    axis : integer, default = None
        Which axis to normalise over, if none, normalise over all axes
    transform: tuple[float, float], default = None
        If transformation values exist already

    Returns
    -------
    tuple[ndarray | Tensor, tuple[float, float] | tuple[ndarray, ndarray] | tuple[Tensor, Tensor]]
        Transformed data & transform values
    """
    if isinstance(data, ndarray):
        data, transform = _ndarray_normalisation(data, mean=mean, axis=axis, transform=transform)
    elif isinstance(data, Tensor):
        data, transform = _tensor_normalisation(data, mean=mean, dim=axis, transform=transform)
    else:
        raise TypeError(f'Type of {type(data)} for data not supported for data normalisation')

    return data, transform


def loader_init(
        dataset: DarkDataset,
        batch_size: int = 120,
        val_frac: float = 0.1,
        idxs: ndarray = None) -> tuple[DataLoader, DataLoader]:
    """
    Initialises training and validation data loaders

    Parameters
    ----------
    dataset : DarkDataset
        Dataset to generate data loaders for
    batch_size : integer, default = 1024
        Number of data inputs per weight update,
        smaller values update the network faster and requires less memory, but is more unstable
    val_frac : float, default = 0.1
        Fraction of validation data
    idxs : ndarray, default = None
        Data indices for random training & validation datasets

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Dataloaders for the training and validation datasets
    """
    kwargs = get_device()[0]

    # Fetch dataset & calculate validation fraction
    val_amount = max(int(len(dataset) * val_frac), 1)

    # If network hasn't trained on data yet, randomly separate training and validation
    if idxs is None or idxs.size != len(dataset):
        # indices = np.random.choice(len(dataset), len(dataset), replace=False)
        idxs = np.arange(len(dataset))
        np.random.shuffle(idxs)

    dataset.idxs = idxs

    train_dataset = Subset(dataset, idxs[:-val_amount])
    val_dataset = Subset(dataset, idxs[-val_amount:])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    if val_frac == 0:
        val_loader = train_loader
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    print(f'\nTraining data size: {len(train_loader.dataset)}'
          f'\tValidation data size: {len(val_loader.dataset)}')

    return train_loader, val_loader
