"""
Loads data and creates data loaders for network training
"""
import pickle
from typing import BinaryIO

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from netloader.utils.utils import get_device
from torchvision.transforms import v2
from numpy import ndarray


class DarkDataset(Dataset):
    """
    A dataset object containing image maps and dark matter cross-sections for PyTorch training

    Attributes
    ----------
    ids : (N) ndarray
        IDs for N clusters in the dataset
    sims : (N) ndarray
        Simulation ID for N clusters in the dataset
    mass : (N) ndarray
        Mass for N clusters in the dataset
    names : (N) ndarray
        Names for N clusters in the dataset
    stellar_frac : (N) ndarray
        Stellar fractions for N clusters in the dataset
    norms : (N,C) ndarray
        Normalisations for C channels of the images for N clusters in the dataset
    labels : (N,1) ndarray | (N,1) Tensor
        Supervised labels for dark matter cross-section for N clusters
    images : (N,C,H,W) ndarray | (N,C,H,W) Tensor
        Lensing, X-ray and stellar maps of height H and width W for N clusters
    aug : Compose
        Image augmentation transform
    idxs : ndarray, default = None
        Data indices for random training & validation datasets
    """
    def __init__(
            self,
            data_dir: str,
            sims: list[str],
            unknown_sims: list[str]):
        """
        Parameters
        ----------
        data_dir : str
            Path to the directory with the cluster datasets
        sims : list[str]
            Which simulations to load that are known
        unknown_sims : list[str]
            Which simulations to load that are unknown
        """
        self._unknown_factor: float = 1e-3
        self.unknown: list[str] = unknown_sims
        self.ids: ndarray = np.array([])
        self.sims: ndarray = np.array([])
        self.mass: ndarray = np.array([])
        self.names: ndarray = np.array([])
        self.norms: ndarray = np.array([])
        self.stellar_frac: ndarray = np.array([])
        self.idxs: ndarray | None = None
        self.images: ndarray | Tensor
        self.labels: ndarray | Tensor = np.array([])
        label: float
        log_0: float = 4e-2
        sim: str
        labels: dict[str, ndarray | list[float]]
        norms_: list[ndarray] = []
        images_: list[ndarray] = []
        file: BinaryIO
        sims_: ndarray
        images: ndarray

        sims_ = np.array(sims + self.unknown)[np.sort(np.unique(
            sims + self.unknown,
            return_index=True,
        )[1])]

        if 'noise' in np.char.lower(sims):
            raise ValueError('Noise cannot be treated as a known simulation')

        for sim in sims_:
            # Generate noise maps
            if sim.lower() == 'noise':
                self._generate_noise(images_, norms_)
                continue

            # Load data from file
            with open(f"{data_dir}{sim.lower().replace('+', '_')}.pkl", 'rb') as file:
                labels, images = pickle.load(file)

            # Remove stellar maps & create labels
            images_.append(images[:, :3])
            norms_.append(labels['norms'])
            label = labels['label'][0]

            # Ensure there are no zero labels
            if label == 0:
                label = log_0

            # Prevent duplicate labels
            while np.isin(
                    [label, label * self._unknown_factor],
                    np.unique(self.labels),
            ).any() and label != 0:
                label *= 0.999

            # Ensure unknown labels are the smallest
            if sim in self.unknown:
                label *= self._unknown_factor

            self.labels = np.concatenate((self.labels, np.ones(len(images)) * label))
            self.sims = np.concatenate((self.sims, [sim] * len(images)))
            self.names = np.concatenate((self.names, [labels['name']] * len(images)))

            # Create IDs
            if 'ClusterID' in labels:
                self.ids = np.concatenate((self.ids, labels['clusterID'].astype(int)))
            else:
                self.ids = np.concatenate((
                    self.ids,
                    np.arange(len(images)) + (np.max(self.ids) + 1 if len(self.ids) > 0 else 0),
                ))

            # Calculate mass and stellar fraction within 100 kpc
            mask = np.where(np.add(*[array ** 2 for array in np.meshgrid(
                np.arange(images.shape[-2]) - images.shape[-2] // 2 + 0.5,
                np.arange(images.shape[-1]) - images.shape[-1] // 2 + 0.5,
            )]) < 25, 1, 0)
            self.mass = np.concatenate((
                self.mass,
                np.sum(images[:, 0] * mask, axis=(-2, -1)) * labels['norms'][:, 0],
            ))

            if images.shape[1] == 3:
                self.stellar_frac = np.concatenate((
                    self.stellar_frac,
                    (np.sum(images[:, -1] * mask, axis=(-2, -1)) * labels['norms'][:, -1] /
                     (np.sum(images[:, 0] * mask, axis=(-2, -1)) * labels['norms'][:, 0])),
                ))
            else:
                self.stellar_frac = np.concatenate((self.stellar_frac, np.zeros(len(images))))


        self.norms = np.concatenate(norms_)
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
        idx : int
            Index of the target cluster

        Returns
        -------
        tuple[ndarray, Tensor, Tensor, ndarray]
            Cluster ID, cluster label, and augmented image map
        """
        return self.ids[idx], self.labels[idx], self.aug(self.images[idx, :2])

    def _generate_noise(self, images: list[ndarray], norms: list[ndarray]) -> None:
        """
        Generates random uniform noise images

        Parameters
        ----------
        images : list[(N,...) ndarray]
            List of dataset images with N images per simulation, requires at least 1 to base the
             noise images shape off
        norms : list[(N,...) ndarray]
            List of normalisations with N normalisations per simulation, requires at least 1 to
             base the noise norms shape off
        """
        if len(images) == 0:
            raise ValueError('Cannot generate noise maps without existing data to base '
                             'them off')

        images.append(np.random.rand(*images[0].shape))
        norms.append(np.ones_like(norms[0]))
        self.labels = np.concatenate((
            self.labels,
            np.ones(len(images[0])) * self._unknown_factor ** 2,
        ))
        self.sims = np.concatenate((self.sims, ['noise'] * len(images[0])))
        self.names = np.concatenate((self.names, ['Noise'] * len(images[0])))
        self.ids = np.concatenate((
            self.ids,
            np.arange(len(images[0])) + (np.max(self.ids) + 1 if len(self.ids) > 0 else 0),
        ))

    def correct_unknowns(self, labels: ndarray) -> ndarray:
        """
        Rescales the unknown labels to their correct values

        Parameters
        ----------
        labels : (N) ndarray
            N labels with unknown values to be corrected

        Returns
        -------
        (N) ndarray
            Corrected labels
        """
        for label in np.unique(labels)[:len(self.unknown)]:
            labels[labels == label] /= self._unknown_factor
        return labels


class GaussianDataset(Dataset):
    """
    A dataset object containing Gaussian toy images and centers for PyTorch training

    Attributes
    ----------
    unknown : int
        Number of unknown classes
    ids : ndarray
        IDs for each Gaussian image in the dataset
    labels : Tensor
        Supervised labels for Gaussian centers
    images : Tensor
        Gaussian images
    idxs : ndarray, default = None
        Data indices for random training & validation datasets
    """
    def __init__(
            self,
            data_path: str,
            known: list[float],
            unknown: list[float],
            names: list[str] | None = None):
        """
        Parameters
        ----------
        data_path : str
            Path to the Gaussian dataset
        known : list[float]
            Known classes and labels
        unknown : list[float]
            Known classes with unknown labels
        names : list[str] | None, default = None
            Names for the simulations in order of known sims first, then unknown sims
        """
        self._unknown_factor: float = 1e-3
        self.unknown: list[float] = np.array(unknown)[~np.in1d(unknown, known)].tolist()
        self.ids: ndarray
        self.names: ndarray
        self.idxs: ndarray | None = None
        self.labels: ndarray | Tensor
        self.images: ndarray | Tensor
        class_: float
        name: str
        bad_idxs: ndarray
        labels: ndarray
        images: ndarray

        with open(data_path, 'rb') as file:
            labels, images = pickle.load(file)

        self.labels = np.round(labels, 5)
        self.images = images

        bad_idxs = ~np.in1d(self.labels, known + self.unknown)
        self.labels = np.delete(self.labels, bad_idxs, axis=0)
        self.images = np.delete(self.images, bad_idxs, axis=0)

        for class_ in self.unknown:
            self.labels[np.in1d(self.labels, class_)] *= self._unknown_factor

        self.ids = np.arange(len(self.labels))
        self.names = self.labels.copy() if names is None else np.array(
            [''] * len(self.labels),
            dtype=np.array(names, dtype=str).dtype,
        )
        self.labels = self.labels[:, np.newaxis]

        if names is not None:
            for name, class_ in zip(names, np.array(known + self.unknown)[np.sort(np.unique(
                    known + self.unknown,
                    return_index=True,
                )[1])]):
                self.names[np.in1d(self.labels, [class_, class_ * self._unknown_factor])] = name

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[ndarray, ndarray | Tensor, ndarray | Tensor]:
        """
        Gets the training data for the given index
        Parameters
        ----------
        idx : int
            Index of the target Gaussian image

        Returns
        -------
        tuple[ndarray, ndarray | Tensor, ndarray | Tensor]
            Image ID, label, image
        """
        return self.ids[idx], self.labels[idx], self.images[idx]

    def correct_unknowns(self, labels: ndarray) -> ndarray:
        """
        Rescales the unknown labels to their correct values

        Parameters
        ----------
        labels : (N) ndarray
            N labels with unknown values to be corrected

        Returns
        -------
        (N) ndarray
            Corrected labels
        """
        for label in np.unique(labels)[:len(self.unknown)]:
            labels[labels == label] /= self._unknown_factor
        return labels


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
    batch_size : int, default = 1024
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
