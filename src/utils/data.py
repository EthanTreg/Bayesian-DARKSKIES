"""
Loads data and creates data loaders for network training
"""
import pickle

import torch
import numpy as np
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from netloader.utils.utils import get_device
from torchvision.transforms import v2


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
    def __init__(self, data_dir: str, sims: list[str], unknown_sims: list[str]):
        """
        Parameters
        ----------
        data_dir : string
            Path to the directory with the cluster datasets
        sims : list[string]
            Which simulations to load
        unknown_sims : list[string]
            Which sims from sims should be unknown, excluding zooms which are already unknown
        """
        self.unknown: list[str] = unknown_sims
        self.ids: ndarray = np.array([])
        self.idxs: ndarray | None = None
        self.images: ndarray | Tensor
        self.labels: ndarray | Tensor = np.array([])
        label: float
        log_0: float = 4e-2
        labels: dict[str, ndarray[float]]
        images: ndarray
        images_: list[ndarray] = []

        for sim in sims:
            # Load data from file
            with open(f"{data_dir}{sim.lower().replace('+', '_')}.pkl", 'rb') as file:
                labels, images = pickle.load(file)

            # Remove stellar maps & create labels
            images_.append(images[:, :2])
            label = labels['label'][0]

            # Ensure unknown labels are the smallest & there are no zero labels
            if sim in self.unknown:
                label = 1e-3
            elif label == 0:
                label = log_0

            # Prevent duplicate labels
            while label in np.unique(self.labels) and label != 0:
                label *= 0.999

            self.labels = np.concatenate((self.labels, np.ones(len(images)) * label))

            # Create IDs
            if 'ClusterID' in labels:
                self.ids = np.concatenate((self.ids, labels['clusterID'].astype(int)))
            else:
                self.ids = np.concatenate((
                    self.ids,
                    np.arange(len(images)) + (np.max(self.ids) + 1 if len(self.ids) > 0 else 0),
                ))

        self.images = np.concatenate(images_)
        self.labels = self.labels[:, np.newaxis]

        # Image augmentations
        self.aug = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=(0, 180)),
        ])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[ndarray, ndarray | Tensor, ndarray | Tensor]:
        """
        Gets the training data for the given index

        Parameters
        ----------
        idx : integer
            Index of the target cluster

        Returns
        -------
        tuple[ndarray, Tensor, Tensor, ndarray]
            Cluster ID, cluster label, and augmented image map
        """
        return self.ids[idx], self.labels[idx], self.aug(self.images[idx])


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
        ----------
        idx : integer
            Index of the target cluster

        Returns
        -------
        tuple[ndarray, Tensor, Tensor]
            Cluster ID, cluster label, cluster latent space
        """
        return self.ids[idx], self.labels[idx], self.latent[idx]


class GaussianDataset(Dataset):
    """
    A dataset object containing Gaussian toy images and centers for PyTorch training

    Attributes
    ----------
    unknown : integer
        Number of unknown classes
    ids : ndarray
        IDs for each Gaussian image in the dataset
    labels : Tensor
        Supervised labels for Gaussian centers
    images : Tensor
        Gaussian images
    transform : tuple[float, float], default = (0, 1)
        Label min and range for 0-1 normalisation
    idxs : ndarray, default = None
        Data indices for random training & validation datasets
    """
    def __init__(self, data_path: str, known: list[float], unknown: list[float]):
        """
        Parameters
        ----------
        data_path : string
            Path to the Gaussian dataset
        known : list[float]
            Known classes and labels
        unknown : list[float]
            Known classes with unknown labels
        """
        self.idxs = None
        self.transform = (0, 1)
        self.unknown = unknown

        with open(data_path, 'rb') as file:
            labels, images = pickle.load(file)

        self.labels = labels
        self.images = images

        bad_idxs = ~np.in1d(self.labels, known + self.unknown)
        self.labels = np.delete(self.labels, bad_idxs, axis=0)
        self.images = np.delete(self.images, bad_idxs, axis=0)

        for i, class_ in enumerate(self.unknown):
            self.labels[np.in1d(self.labels, class_)] = 10 ** -(i + 3)

        self.ids = np.arange(len(self.labels))
        self.labels = np.log10(self.labels)
        self.labels = torch.from_numpy(self.labels).float()[:, None]
        self.images = torch.from_numpy(self.images).float()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[ndarray, Tensor, Tensor]:
        """
        Gets the training data for the given index
        Parameters
        ----------
        idx : integer
            Index of the target Gaussian image

        Returns
        -------
        tuple[ndarray, Tensor, Tensor]
            Image ID, label, image
        """
        return self.ids[idx], self.labels[idx], self.images[idx]


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
