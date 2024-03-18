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
    labels : Tensor
        Supervised labels for dark matter cross-section for each cluster
    images : Tensor
        Lensing and X-ray maps for each cluster
    meta : Tensor
        Metadata for each cluster in the dataset
    aug : Compose
        Image augmentation transform
    transform : tuple[float, float], default = (0, 1)
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
        self.idxs = None
        self.transform = (0, 1)
        log_0 = 5e-2

        # Load data from file
        with open(data_path, 'rb') as file:
            labels, images = pickle.load(file)

        # Get specified sim data
        idxs = np.in1d(labels['sim'], sims)

        # Remove stellar maps & create labels & IDs
        self.images = np.moveaxis(images[idxs, ..., :2], 3, 1)
        del images
        self.labels = labels['label'][idxs]
        self.ids = labels['clusterID'][idxs]

        # Transform labels
        if 'CDM_low+baryons' in sims:
            self.labels[np.in1d(labels['sim'], 'CDM_low+baryons')[idxs]] = 4.9e-2

        if 'CDM_hi+baryons' in sims:
            self.labels[np.in1d(labels['sim'], 'CDM_hi+baryons')[idxs]] = 5.1e-2

        if 'vdSIDM+baryons' in sims:
            vd_idxs = np.in1d(labels['sim'], 'vdSIDM+baryons')[idxs]

            for idx, sigma in zip(
                    np.array_split(np.argsort(self.labels[vd_idxs]), 3),
                    [1e-6, 1e-5, 1e-4],
            ):
                self.labels[np.flatnonzero(vd_idxs)[idx]] = sigma

        self.labels[self.labels == 0] = log_0
        self.labels = np.log10(self.labels)

        # Metadata
        del labels['galaxy_catalogues']
        del labels['sim']
        del labels['clusterID']
        del labels['label']
        self.meta = torch.from_numpy(np.array(
            list(labels.values()),
            dtype=float,
        )).swapaxes(0, 1)[idxs]
        self.meta, self.meta_transform = data_normalisation(self.meta, axis=0)

        self.labels = torch.from_numpy(self.labels).float()[:, None]
        self.images = torch.from_numpy(self.images).float()
        self.images = torch.cat((
            self.images.reshape(len(self.images), -1),
            self.meta.float(),
        ), dim=1)

        # Image augmentations
        self.aug = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=(0, 180)),
        ])

    def __len__(self) -> int:
        return len(self.ids)

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
        image = self.images[idx, :-self.meta.size(-1)]
        image = torch.cat((
            self.aug(image.view(-1, 100, 100)).flatten(),
            self.images[idx, -self.meta.size(-1):],
        ))
        return self.ids[idx], self.labels[idx], image, self.meta[idx]

    def normalise(self, idxs: list[int] | Tensor = None, transform: tuple[float, float] = None):
        """
        Normalises the labels

        Parameters
        ----------
        idxs : list[integer] | (N) Tensor
            Which indices should be normalised, where N is either the indices to keep, or equal to
            the number of labels and is a boolean tensor
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


class ClusterDataset(Dataset):
    """
    A dataset object containing a cluster latent space and dark matter cross-sections for PyTorch
    training

    Attributes
    ----------
    ids : ndarray
        IDs for each cluster in the dataset
    labels : Tensor
        Supervised labels for dark matter cross-section for each cluster
    latent : Tensor
        Cluster latent space for each cluster
    transform : tuple[float, float], default = (0, 1)
        Label min and range for 0-1 normalisation
    idxs : ndarray, default = None
        Data indices for random training & validation datasets
    """
    def __init__(self, data_path: str, sigmas: list[str]):
        """
        Parameters
        ----------
        data_path : string
            Path to the data file with the cluster latent space
        sigmas : list[string]
            Cross-sections to use from the dataset
        """
        self.transform = (0, 1)
        bad_keys = []
        self.idxs = None

        with open(data_path, 'rb') as file:
            data = pickle.load(file)

        # Remove cross-sections not specified
        for key in data.keys():
            if key not in sigmas:
                bad_keys.append(key)

        for bad_key in bad_keys:
            del data[bad_key]

        data = np.concatenate(list(data.values()))

        self.ids = data[:, 0].astype(int)
        self.labels = torch.from_numpy(data[:, 1:2]).float()
        self.latent = torch.from_numpy(data[:, -7:]).float()
        self.idxs = np.append(
            np.argwhere(self.labels.flatten() != torch.unique(self.labels)[2]).flatten(),
            np.argwhere(self.labels.flatten() == torch.unique(self.labels)[2]).flatten(),
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[ndarray, Tensor, Tensor]:
        """
        Gets the training data for the given index

        Parameters
        idx : integer
            Index of the target cluster

        Returns
        -------
        tuple[ndarray, Tensor, Tensor]
            Cluster ID, cluster label, cluster latent space
        """
        return self.ids[idx], self.labels[idx], self.latent[idx]


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
