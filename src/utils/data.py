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
    indices : ndarray
        Data indices for random training & validation datasets
    labels : Tensor
        Supervised labels for dark matter cross-section for each cluster
    images : Tensor
        Lensing and X-ray maps for each cluster
    transform : Compose
        Image augmentation transform
    """
    def __init__(self, data_path: str):
        """
        Parameters
        ----------
        data_path : string
            Path to the data file with the cluster dataset
        """
        idxs = []
        sims = ['CDM+baryons', 'SIDM0.1+baryons', 'SIDM1+baryons']
        self.indices = None

        # Load data from file
        with open(data_path, 'rb') as file:
            labels, images = pickle.load(file)

        # Get specified sim data
        for sim in sims:
            idxs.extend(np.argwhere(np.array(labels['sim']) == sim).flatten())

        # Remove stellar maps
        self.images = np.moveaxis(np.delete(images, -1, axis=-1), 3, 1)[idxs]

        # Change labels to one-hot class labels
        self.labels = np.array(labels['label'])[idxs]
        classes = np.unique(self.labels)
        mapping = dict(zip(classes, np.arange(classes.size)))
        self.labels = np.vectorize(lambda x: mapping[x])(self.labels)

        # Uses cluster IDs if provided, otherwise, number dataset in order
        if 'clusterID' in labels:
            self.ids = np.array(labels['clusterID'])[idxs]
        else:
            self.ids = np.arange(self.images.shape[0])

        self.labels = torch.from_numpy(self.labels)[:, None]
        self.images = torch.from_numpy(self.images).float()

        # Image augmentations
        self.transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=(0, 180)),
        ])

    def __len__(self) -> int:
        return self.images.shape[0]

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
            Cluster ID, dark matter cross-section, and augmented image map
        """
        return self.ids[idx], self.labels[idx], self.transform(self.images[idx])


def _balance_data(labels: ndarray, data: list[ndarray]) -> tuple[ndarray, list[ndarray]]:
    """
    Balances training data so that there is an equal amount of each class
    
    Parameters
    ----------
    labels : ndarray
        Classification labels to balance
    data : list[ndarray]
        Corresponding datasets to balance based off labels
    
    Returns
    -------
    
    """
    idxs = []

    # Calculate the number of each class
    classes, class_counts = np.unique(labels, return_counts=True)
    class_diffs = class_counts - np.min(class_counts)

    # Find indices that have an equal amount of each class
    for class_value, class_diff in zip(classes, class_diffs):
        idxs.extend(np.argwhere(labels == class_value)[class_diff:, 0])

    return labels[idxs], [dataset[idxs] for dataset in data]


def data_init(
        data_path: str,
        batch_size: int = 120,
        val_frac: float = 0.1,
        indices: ndarray = None) -> tuple[DataLoader, DataLoader]:
    """
    Initialises training and validation datasets

    Parameters
    ----------
    data_path : string
        Path to the dataset
    batch_size : integer, default = 1024
        Number of data inputs per weight update,
        smaller values update the network faster and requires less memory, but is more unstable
    val_frac : float, default = 0.1
        Fraction of validation data
    indices : ndarray, default = None
        Data indices for random training & validation datasets

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Dataloaders for the training and validation datasets
    """
    kwargs = get_device()[0]

    # Fetch dataset & calculate validation fraction
    dataset = DarkDataset(data_path)
    val_amount = max(int(len(dataset) * val_frac), 1)

    # If network hasn't trained on data yet, randomly separate training and validation
    if indices is None or indices.size != len(dataset):
        # indices = np.random.choice(len(dataset), len(dataset), replace=False)
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)

    dataset.indices = indices

    train_dataset = Subset(dataset, indices[:-val_amount])
    val_dataset = Subset(dataset, indices[-val_amount:])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    if val_frac == 0:
        val_loader = train_loader
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    print(f'\nTraining data size: {len(train_dataset)}\tValidation data size: {len(val_dataset)}')

    return train_loader, val_loader
