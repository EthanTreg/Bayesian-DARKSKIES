"""
Loads data and creates data loaders for network training
"""
import os
import pickle
from types import ModuleType
from typing import BinaryIO, Any

import torch
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from netloader.utils.types import ArrayLike
from netloader.data import BaseDataset, DataList
from pandas import DataFrame
from numpy import ndarray

from src.utils.utils import ROOT


class DarkDataset(BaseDataset):
    """
    A dataset object containing image maps and dark matter cross-sections for PyTorch training.

    Attributes
    ----------
    unknown : list[str]
        List of unknown simulations
    idxs : ndarray
        Index for each sample in the dataset with shape (N) and type int
    low_dim : ndarray | Tensor | None, default = None
        Supervised labels for the dark matter cross-section with shape (N) and type float, where N
        is the number of samples
    high_dim : ndarray | Tensor | object
        Total mass, X-ray, and stellar mass maps with shape (N,C,H,W) and type float, with number of
        channels C, height H, and width W
    extra : DataFrame
        Additional data for each sample in the dataset of length N with shape (N,...) and type Any,
        contains the sample ids (ids), simulations (sims), simulation name (names), and image
        normalisations (norms)
    aug : Compose
        Image augmentation transform

    Methods
    -------
    get_high_dim(idx) -> ndarray | Tensor
        Gets a high dimensional sample of the given index
    get_extra(idx) -> ndarray | Tensor
        Gets extra data for the sample of the given index
    unique_labels(labels, classes, factor=0.999) -> ndarray
        Ensures that all simulations have a unique label
    correct_unknowns(labels) -> ndarray:
        Rescales the unknown labels to their correct values
    """
    _unknown_factor: float = 1e-3
    _keys: tuple[str, ...] = ('ids', 'sims', 'names', 'norms', 'mass', 'bcg_e')

    def __init__(
            self,
            data_dir: str,
            sims: list[str],
            unknown_sims: list[str],
            mix: bool = False,
            cdm_sigma: float = 1e-2) -> None:
        """
        Parameters
        ----------
        data_dir : str
            Path to the directory with the cluster datasets
        sims : list[str]
            Which simulations to load that are known
        unknown_sims : list[str]
            Which simulations to load that are unknown
        mix : bool, default = False
            If Mix-Up should be used for training
        cdm_sigma : float, default = 1e-2
            Effective cross-section for CDM simulations
        """
        super().__init__()
        label: float
        sim: str
        labels: dict[str, ndarray]
        sims_: ndarray
        images: ndarray
        extra: DataFrame
        self._mix: bool = mix
        self._idx: int = -1
        self.unknown: list[str] = unknown_sims
        self.aug: v2.Transform
        self.extra: DataFrame = DataFrame([])

        sims_ = np.array(sims + self.unknown)[np.sort(np.unique(
            sims + self.unknown,
            return_index=True,
        )[1])]

        if 'noise' in np.char.lower(sims):
            raise ValueError('Noise cannot be treated as a known simulation')

        for sim in sims_:
            if sim.lower() == 'noise' and self.high_dim is not None:
                labels, images, extra = self._generate_noise([int(1e3), *self.high_dim.shape[1:]])
            elif sim.lower() == 'noise':
                raise ValueError('Cannot generate noise maps without existing data to base them '
                                 'off')
            else:
                labels, images, extra = self._load_data(sim, data_dir)

            label = float(labels['label'][0])

            if 'flamingo' in sim.lower() or 'cdm' in sim.lower():
                label = cdm_sigma

            # Ensure there are no zero labels
            if label == 0:
                label = cdm_sigma

            # Ensure unknown labels are the smallest
            if sim in self.unknown:
                label *= self._unknown_factor

            if self.high_dim is None:
                self.high_dim = images[:, :3]
                self.low_dim = np.ones(len(images)) * label
            else:
                self.high_dim = np.concat((self.high_dim, images[:, :3]), axis=0)
                self.low_dim = np.concat((self.low_dim, np.ones(len(images)) * label), axis=0)

            self.extra = pd.concat([self.extra, extra], axis=0)

        self.extra.columns = self._keys
        self.low_dim = self.low_dim[:, np.newaxis]

        # Image augmentations
        self.aug = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=(0, 180)),
        ])

    def __getitem__(
            self,
            idx: int) -> tuple[int, ndarray | Tensor, ndarray | Tensor, Any]:
        idxs: ndarray | Tensor

        if self._mix and np.random.choice([True, False]):
            idxs = (self.low_dim != self.low_dim[idx]).flatten()
            self._idx = np.random.choice(self.idxs[~np.isin(
                self.extra['sims'],
                self.unknown,
            ) & idxs if isinstance(idxs, ndarray) else idxs.numpy()])
        elif self._mix:
            self._idx = -1
        return super().__getitem__(idx)

    @staticmethod
    def _load_data(sim: str, data_dir: str) -> tuple[dict[str, ndarray], ndarray, DataFrame]:
        """
        Loads data from a file

        Parameters
        ----------
        sim : str
            Simulation name
        data_dir : str
            Path to the directory of simulation data

        Returns
        -------
        tuple[dict[str, ndarray], ndarray, DataFrame]
            Labels, images, extra data
        """
        labels: dict[str, ndarray]
        file: BinaryIO
        images: ndarray
        extra: DataFrame

        with open(os.path.join(
            ROOT,
            data_dir,
            f"{sim.lower().replace('+', '_')}.pkl",
        ), 'rb') as file:
            labels, images = pickle.load(file)

        if images.shape[1] == 4:
            images = np.delete(images, 1, axis=1)
            labels['norms'] = np.delete(labels['norms'], 1, axis=1)

        extra = DataFrame(np.array([
            f'{sim}_' +  np.arange(len(images)).astype(str),
            [sim] * len(images),
            [labels['name']] * len(images),
            labels['norms'].tolist(),
            labels.pop('mass', np.zeros(len(images))).tolist(),
            np.stack((
                labels.pop('BCG_e1', np.zeros(len(images))),
                labels.pop('BCG_e2', np.zeros(len(images))),
            ), axis=1).tolist(),
        ], dtype=object).swapaxes(0, 1))
        return labels, images, extra

    def _generate_noise(self, shape: list[int]) -> tuple[dict[str, ndarray], ndarray, DataFrame]:
        """
        Generates random uniform noise images

        Parameters
        ----------
        shape : list[int]
            Shape of the noise images to generate

        Returns
        -------
        tuple[dict[str, ndarray], ndarray, DataFrame]
            Labels, images, extra data
        """
        images: ndarray = np.random.rand(*shape)
        labels: ndarray = np.ones(shape[0]) * self._unknown_factor
        extra: DataFrame = DataFrame(np.array([
            'noise_' + np.arange(shape[0]).astype(str),
            ['noise'] * shape[0],
            ['Noise'] * shape[0],
            np.ones(shape[:2]).tolist(),
        ], dtype=object).swapaxes(0, 1))
        return {'label': labels}, images, extra

    def get_low_dim(self, idx: int) -> ndarray | Tensor:
        module: ModuleType

        if self._mix and self._idx >= 0 and self.extra['sims'].iloc[idx] not in self.unknown:
            module = np if isinstance(self.high_dim, ndarray) else torch
            return module.log10(module.mean(
                10 ** module.stack([self.low_dim[idx], self.low_dim[self._idx]]),
                **{'axis' if isinstance(self.low_dim, ndarray) else 'dim': 0},
            ))
        return self.low_dim[idx]

    def get_high_dim(self, idx: int) -> ndarray | Tensor:
        module: ModuleType

        if self._mix and self._idx >= 0 and self.extra['sims'].iloc[idx] not in self.unknown:
            module = np if isinstance(self.high_dim, ndarray) else torch
            return self.aug(module.mean(
                module.stack([self.high_dim[idx], self.high_dim[self._idx]]),
                **{'axis' if isinstance(self.high_dim, ndarray) else 'dim': 0},
            ))
        return self.aug(self.high_dim[idx])

    def get_extra(self, idx: int) -> list[Any]:
        return self.extra.iloc[idx].tolist()

    def unique_labels(self, labels: ndarray, classes: ndarray, factor: float = 0.999) -> ndarray:
        """
        Ensures that all simulations have a unique label

        Parameters
        ----------
        labels : ndarray
            Labels, of shape (N), to make unique depending on classes
        classes : ndarray
            Classes, of shape (N), that each label belongs to
        factor : float, default = 0.999
            Factor to scale the label by until it is unique

        Returns
        -------
        ndarray
            Labels, of shape (N), with unique labels depending on their class
        """
        idx : int
        class_: int | float | str
        label: float
        idxs: ndarray
        new_labels: ndarray = np.zeros_like(labels)

        for class_, idx in zip(*np.unique(classes, return_index=True)):
            idxs = classes == class_
            label = labels[idx]

            while np.isin(
                    [label, label * self._unknown_factor],
                    new_labels,
            ).any() and label != 0:
                label *= factor

            new_labels[idxs] = label
        return new_labels

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
        label: float

        for label in np.unique(labels)[:len(self.unknown)]:
            labels[labels == label] /= self._unknown_factor
        return labels


class DarkPointDataset(BaseDataset):
    """
    A dataset object containing shear catalogues and dark matter cross-sections for PyTorch
    training.
    """
    _unknown_factor: float = 1e-3
    _keys: tuple[str, ...] = ('ids', 'sims', 'names', 'norms')

    def __init__(
            self,
            data_dir: str,
            sims: list[str],
            unknown_sims: list[str],
            spatial_num: int = 2,
            cdm_sigma: float = 1e-2) -> None:
        """
        Parameters
        ----------
        data_dir : str
            Path to the directory with the cluster datasets
        sims : list[str]
            Which simulations to load that are known
        unknown_sims : list[str]
            Which simulations to load that are unknown
        mix : bool, default = False
            If Mix-Up should be used for training
        cdm_sigma : float, default = 1e-2
            Effective cross-section for CDM simulations
        """
        super().__init__()
        label: float
        sim: str
        labels: dict[str, ndarray]
        sims_: ndarray
        points: ndarray
        extra: DataFrame
        self._idx: int = -1
        self.num: int = spatial_num
        self.unknown: list[str] = unknown_sims
        self.aug: v2.Transform
        self.extra: DataFrame = DataFrame([])
        self.high_dim: list[ndarray] = []

        sims_ = np.array(sims + self.unknown)[np.sort(np.unique(
            sims + self.unknown,
            return_index=True,
        )[1])]

        if 'noise' in np.char.lower(sims):
            raise ValueError('Noise cannot be treated as a known simulation')

        for sim in sims_:
            labels, points, extra = self._load_data(sim, data_dir)
            label = float(labels['label'][0])
            self.high_dim.extend(points)

            if 'flamingo' in sim.lower() or 'cdm' in sim.lower():
                label = cdm_sigma

            # Ensure there are no zero labels
            if label == 0:
                label = cdm_sigma

            # Ensure unknown labels are the smallest
            if sim in self.unknown:
                label *= self._unknown_factor

            if self.low_dim is None:
                self.low_dim = np.ones(len(points)) * label
            else:
                self.low_dim = np.concat((self.low_dim, np.ones(len(points)) * label), axis=0)

            self.extra = pd.concat([self.extra, extra], axis=0)

        self.extra.columns = self._keys
        self.low_dim = self.low_dim[:, np.newaxis]

    @staticmethod
    def _load_data(sim: str, data_dir: str) -> tuple[dict[str, ndarray], list[ndarray], DataFrame]:
        """
        Loads data from a file

        Parameters
        ----------
        sim : str
            Simulation name
        data_dir : str
            Path to the directory of simulation data

        Returns
        -------
        tuple[dict[str, ndarray], ndarray, DataFrame]
            Labels, images, extra data
        """
        points: list[ndarray]
        labels: dict[str, ndarray]
        file: BinaryIO
        extra: DataFrame

        with open(os.path.join(
            ROOT,
            data_dir,
            f"{sim.lower().replace('+', '_')}.pkl",
        ), 'rb') as file:
            labels, points = pickle.load(file)

        if 'flamingo' in sim.lower():
            labels['norms'] = np.delete(labels['norms'], 1, axis=1)

        labels['norms'] = labels['norms'][:, :3]
        extra = DataFrame(np.array([
            f'{sim}_' +  np.arange(len(points)).astype(str),
            [sim] * len(points),
            [labels['name']] * len(points),
            labels['norms'].tolist(),
        ], dtype=object).swapaxes(0, 1))
        return labels, points, extra

    @staticmethod
    def collate(
            items: list[tuple[int, Tensor, DataList[Tensor]]],
    ) -> tuple[ndarray, Tensor, list[DataList[Tensor]]]:
        """
        Collates a list of items into a batch, ensuring all point clouds have the same number of
        points by repeating random points for smaller point clouds.

        Parameters
        ----------
        items : list[tuple[int, Tensor, DataList[Tensor]]]
            List of items to collate, where each item is a tuple containing an index, a label with
            shape (1) and type float, and a point cloud with shapes (N,C) and optionally (N,D) and
            type float, where N is the number of points, C is the number of input coordinates, and D
            is the number of input features, the first tensor must be the point coordinates with an
            optional second tensor with point features, each point cloud can have a different N

        Returns
        -------
        tuple[ndarray, Tensor, DataList[Tensor]]
            Collated indices with shape (B) and type int, labels with shape (B,1) and type float,
            and point clouds with shape (B,N,C) and optionally shape (B,N,D) and type float, where B
            is the batch size and N is the maximum number of points
        """
        num: int
        repeat_num: int
        points: list[DataList[Tensor]] = []
        idxs: tuple[int, ...]
        labels: tuple[Tensor, ...]
        data: tuple[DataList[Tensor], ...]
        sub_point: Tensor
        point: DataList[Tensor]

        idxs, labels, data, _ = zip(*items)
        num = max(point.len(False) for point in data)

        for point in data:
            if point.len(False) == num:
                points.append(DataList([sub_point[None] for sub_point in point]))
                continue

            repeat_num = num - point.len(False)
            points.append(DataList([torch.cat((
                sub_point,
                sub_point[torch.randint(0, len(sub_point), (repeat_num,))],
            ))[None] for sub_point in point]))

        # return np.stack(idxs), torch.stack(labels), DataList.collate(points)
        return np.stack(idxs), torch.stack(labels), points

    def get_high_dim(self, idx: int) -> DataList[ArrayLike]:
        return DataList([
            self.high_dim[idx][:, :self.num],
            self.high_dim[idx][:, self.num:],
        ])

    def get_extra(self, idx: int) -> list[Any]:
        return self.extra.iloc[idx].tolist()

    def unique_labels(self, labels: ndarray, classes: ndarray, factor: float = 0.999) -> ndarray:
        """
        Ensures that all simulations have a unique label

        Parameters
        ----------
        labels : ndarray
            Labels, of shape (N), to make unique depending on classes
        classes : ndarray
            Classes, of shape (N), that each label belongs to
        factor : float, default = 0.999
            Factor to scale the label by until it is unique

        Returns
        -------
        ndarray
            Labels, of shape (N), with unique labels depending on their class
        """
        idx : int
        class_: int | float | str
        label: float
        idxs: ndarray
        new_labels: ndarray = np.zeros_like(labels)

        for class_, idx in zip(*np.unique(classes, return_index=True)):
            idxs = classes == class_
            label = labels[idx]

            while np.isin(
                    [label, label * self._unknown_factor],
                    new_labels,
            ).any() and label != 0:
                label *= factor

            new_labels[idxs] = label
        return new_labels

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
        label: float

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

        with open(os.path.join(ROOT, data_path), 'rb') as file:
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
