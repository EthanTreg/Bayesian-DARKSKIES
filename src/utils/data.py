"""
Loads data and creates data loaders for network training
"""
import pickle

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset

from src.utils.utils import get_device


class DarkDataset(Dataset):
    """
    A dataset object containing image maps and dark matter cross-sections for PyTorch training

    Attributes
    ----------
    ids : list[integer]
        IDs for each cluster in the dataset
    indices : ndarray
        Data indices for random training & validation datasets
    labels : Tensor
        Supervised labels for dark matter cross-section for each cluster
    """
    def __init__(self, data_path: str):
        """
        Parameters
        ----------
        data_path : string
            Path to the data file with the cluster dataset
        """
        self.indices = None

        # Load data from file
        with open(data_path, 'rb') as file:
            params, images = pickle.load(file)

        self.images = torch.from_numpy(np.delete(images, -1, axis=-1))
        self.labels = torch.tensor(params['label'])

        # Uses cluster IDs if provided, otherwise, number dataset in order
        if 'clusterID' in params:
            self.ids = params['clusterID']
        else:
            self.ids = np.arange(self.images.shape[0])

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[np.ndarray, Tensor, Tensor]:
        """
        Gets the training data for the given index

        Parameters
        ----------
        idx : integer
            Index of the target cluster

        Returns
        -------
        tuple[ndarray, Tensor, Tensor]
            Cluster ID, dark matter cross-section, and image map
        """
        return self.ids[idx], self.labels[idx], self.images[idx]


def data_init(
        data_path: str,
        batch_size: int = 120,
        val_frac: float = 0.1,
        indices: np.ndarray = None) -> tuple[DataLoader, DataLoader]:
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
        indices = np.random.choice(len(dataset), len(dataset), replace=False)

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
