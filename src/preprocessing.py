"""
Takes all pickle files in a directory and subdirectories of labels and images and combines them into
a single pickle file
"""
import os
import pickle
from typing import BinaryIO
from logging import getLogger

import numpy as np
from numpy import ndarray

# Required to initialise the logger
from src.utils.utils import dict_list_append, list_dict_convert


def load_pickle(path: str) -> tuple[
    dict[str, float | list[float] | list[ndarray]],
    list[ndarray] | ndarray
]:
    """
    Loads labels and images from pickle file and finds which is the labels and which is the images

    Parameters
    ----------
    path : str
        Path to the pickle file

    Returns
    -------
    tuple[dict[str, list[float] | list[ndarray]], list[ndarray] | ndarray]
        Labels and images
    """
    data: (tuple[dict[str, float | list[float] | list[ndarray]], list[ndarray] | ndarray] |
           tuple[list[ndarray] | ndarray, dict[str, list[float] | list[ndarray]]])
    file: BinaryIO

    with open(path, 'rb') as file:
        data = pickle.load(file)

    if isinstance(data[0], dict):
        return data

    return data[1], data[0]


def main(dir_path: str, overwrite: bool = False, name: str = '', save_path: str = '') -> None:
    """
    Takes all pickle files in a directory and subdirectories of labels and images and combines them
    into a single pickle file

    Parameters
    ----------
    dir_path : str
        Path to the directory
    overwrite : bool, default = False
        If a file with the same save name already exists, should it be overwritten
    name : str, default = ''
        Name of the data
    save_path : str, default = ''
        Path to save the combined pickle file, if empty, will not save
    """
    root: str
    path: str
    files: list[str]
    value: list[float] | list[ndarray]
    image: list[ndarray] | ndarray
    images: list[ndarray] | ndarray = []
    label: (dict[str, float | list[float] | list[ndarray] | ndarray] |
            list[dict[str, float | ndarray]])
    labels: dict[str, str | list[float | None] | list[ndarray] | ndarray] = {}
    file: BinaryIO

    # Loop through all files in the directory and subdirectories
    for root, _, files in os.walk(dir_path):
        for path in files:
            label, image = load_pickle(os.path.join(root, path))

            # If the label is a list of dictionaries, convert to dictionary of lists
            if isinstance(label, list):
                label = list_dict_convert(label)

            # Merge props dictionary into label dictionary
            if 'props' in label and isinstance(label['props'], list):
                label = label | list_dict_convert(label['props'])
                del label['props']
            elif 'props' in label:
                label = label | label['props']
                del label['props']

            # Merge label into all labels
            dict_list_append(labels, label)

            # Merge image into all images
            if np.ndim(image) == 4:
                images.extend(image)
            else:
                images.append(image)

    # Convert lists to numpy arrays
    for key, value in labels.items():
        labels[key] = np.array(value, dtype=object if key == 'galaxy_catalogues' else None)

    # Make sure shape is (N,C,H,W)
    images = np.array(images)

    if images.shape[-1] != images.shape[-2]:
        images = np.moveaxis(images, -1, 1)

    # Normalise images and save normalisations in labels
    labels['name'] = name
    labels['norms'] = np.max(images, axis=(-2, -1))
    images /= labels['norms'][..., np.newaxis, np.newaxis]

    if not save_path:
        return

    if not overwrite and os.path.exists(save_path):
        getLogger(__name__).error(f'{save_path} already exists and overwrite is False, file will '
                                  f'not be saved')
        return

    with open(save_path, 'wb') as file:
        pickle.dump((labels, images), file)


if __name__ == "__main__":
    main(
        '../data/darkskies/darkskies_new/CDM/',
        overwrite=True,
        name='DARKSKIES-0',
        save_path='../data/darkskies_cdm.pkl',
    )
