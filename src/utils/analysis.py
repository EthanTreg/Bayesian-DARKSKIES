"""
Functions to analyse the neural network
"""
import os
import pickle
from typing import Any, BinaryIO

import numpy as np
from numpy import ndarray


def _red_chi_acc(
        dof: int,
        values: ndarray,
        target: ndarray,
        errors: ndarray) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Calculates the reduced chi square and mean squared error along the last dimension

    Parameters
    ----------
    values : (...,L) ndarray
        L predicted values
    target : (...,L) ndarray
        L target values
    errors : (...,L) ndarray
        L uncertainties

    Returns
    -------
    tuple[(...) ndarray, (...) ndarray, (...) ndarray, (...) ndarray]
        Reduced chi square, reduced chi square uncertainty, mean squared error, and mean squared
        error uncertainty
    """
    red_chi = np.sum(((values - target) / errors) ** 2, axis=-1) / dof
    red_chi_error = 2 * np.sqrt(red_chi / dof)
    acc = np.mean((values - target) ** 2, axis=-1)
    acc_error = 2 * np.sqrt(np.sum(((values - target) * errors) ** 2, axis=-1)) / values.shape[-1]
    return red_chi, red_chi_error, acc, acc_error


def batch_train_summary(
        num: int,
        dir_: str,
        idx: int | None = None) -> dict[str, list[str] | list[ndarray] | ndarray]:
    """
    Generates a summary of batch training by averaging over the number of repeats

    Parameters
    ----------
    num : int
        Batch train number
    dir_ : str
        Directory of saved batch training data
    idx : int | None, default = None
        If a specific simulation should be indexed

    Returns
    -------
    dict[str, list[str] | list[(S) ndarray] | (N,S) ndarray]
        Batch train summary, where S is the number of simulations and N is the number of runs
    """
    value: list[ndarray] | dict[str, ndarray]
    post_data: dict[str, list[str] | list[ndarray] | ndarray] = {
        'means': [],
        'weighted_means': [],
        'stds': [],
        'stes': [],
        'errors': [],
        'description': [],
        'sims': [],
        'unknown_sims': []
    }

    with open(os.path.join(dir_, f'batch_train_{num}.pkl'), 'rb') as file:
        data = pickle.load(file)

    for value in data.values():
        post_data['means'].append(np.mean(value['means'], axis=0))
        post_data['weighted_means'].append(np.average(
            value['means'],
            weights=value['stes'] ** -2,
            axis=0,
        ))
        post_data['stds'].append(np.std(value['means'], axis=0, ddof=1))
        post_data['stes'].append(post_data['stds'][-1] / np.sqrt(len(value['means'])))
        post_data['errors'].append(np.sqrt(np.sum(value['stes'] ** -2, axis=0) ** -1))
        post_data['sims'].append(value['sims'])

        if 'description' in value:
            post_data['description'].append(value['description'])

        if 'unknown_sims' in value:
            post_data['unknown_sims'].append(value['unknown_sims'])

    for key, value in post_data.items():
        if key not in {'description', 'sims', 'unknown_sims'}:
            try:
                if idx is not None:
                    value = [val[idx] for val in value]

                post_data[key] = np.stack(value)
            except ValueError:
                pass
    return post_data


def multi_batch_train_summary(
        key_idx: int,
        dir_: str,
        range_: tuple[int, int],
        idxs: list[int] | None = None,
        target: ndarray | None = None) -> dict[str, list[str] | list[ndarray] | ndarray]:
    """
    Combines multiple batch training summaries by averaging over the number of repeats

    Parameters
    ----------
    key_idx : int
        Which run to index
    dir_ : str
        Directory of saved batch training data
    range_ : tuple[int, int]
        Range of batch training data numbers
    idxs : list[int] | None, default = None
        Which simulations to calculate the reduced chi square and mean squared error for
    target : (S) ndarray | None, default = None
        Target values for S simulations

    Returns
    -------
    dict[str, list[str] | list[(S) ndarray] | (N,S) ndarray]
        Batch training summaries, where S is the number of simulations and N is the number of runs
    """
    i: int
    key: str
    value: list[ndarray]
    post_data: dict[str, list[str] | list[ndarray] | ndarray] = {
        'means': [],
        'weighted_means': [],
        'stds': [],
        'stes': [],
        'errors': [],
        'log_means': [],
        'log_weighted_means': [],
        'log_stds': [],
        'log_stes': [],
        'log_errors': [],
        'targets': [],
        'description': [],
        'sims': [],
        'unknown_sims': [],
        'nets': [],
    }

    for i in range(*range_):
        with open(os.path.join(dir_, f'batch_train_{i}.pkl'), 'rb') as file:
            data = pickle.load(file)

        batch_data = data[list(data.keys())[key_idx]]
        post_data['means'].append(np.mean(batch_data['means'], axis=0))
        post_data['weighted_means'].append(np.average(
            batch_data['means'],
            weights=batch_data['stes'] ** -2,
            axis=0,
        ))
        post_data['stds'].append(np.std(batch_data['means'], axis=0, ddof=1))
        post_data['stes'].append(post_data['stds'][-1] / np.sqrt(len(batch_data['means'])))
        post_data['errors'].append(np.sqrt(np.sum(batch_data['stes'] ** -2, axis=0) ** -1))

        post_data['log_means'].append(np.mean(batch_data['log_means'], axis=0))
        post_data['log_weighted_means'].append(np.average(
            batch_data['log_means'],
            weights=batch_data['log_stes'] ** -2,
            axis=0,
        ))
        post_data['log_stds'].append(np.std(batch_data['log_means'], axis=0, ddof=1))
        post_data['log_stes'].append(
            post_data['log_stds'][-1] / np.sqrt(len(batch_data['log_means'])),
        )
        post_data['log_errors'].append(np.sqrt(np.sum(batch_data['log_stes'] ** -2, axis=0) ** -1))

        post_data['targets'].append(batch_data['targets'])
        post_data['description'].append(
            batch_data['description'] if 'description' in batch_data else '',
        )
        post_data['sims'].append(batch_data['sims'])
        post_data['nets'].append(batch_data['nets'] if 'nets' in batch_data else None)

        if 'unknown_sims' in batch_data:
            post_data['unknown_sims'].append(batch_data['unknown_sims'])

    for key, value in post_data.items():
        if key not in {'description', 'sims', 'unknown_sims'}:
            try:
                post_data[key] = np.stack(value)
            except ValueError:
                pass

    if target is not None:
        assert isinstance(post_data['means'], ndarray)
        assert isinstance(post_data['stes'], ndarray)
        (post_data['red_chi'],
         post_data['red_chi_error'],
         post_data['acc'],
         post_data['acc_error']) = _red_chi_acc(
            post_data['means'].shape[-1],
            post_data['means'][:, idxs or slice(idxs)],
            target[idxs or slice(idxs)],
            post_data['stes'][:, idxs or slice(idxs)],
        )
    return post_data


def hyperparam_summary(path: str) -> tuple[list[int], ndarray, ndarray]:
    """
    Returns the accuracy and latent loss with standard deviations from the hyperparam_search results

    Parameters
    ----------
    path : str
        Path to the hyperparameter search results

    Returns
    -------
    tuple[list[int], (Nd,Ns,2) ndarray, (Nd,Ns,2) ndarray]
        List of latent dimensions, mean accuracy and latent loss, and standard deviations for Nd
        latent dimension tests and Ns simulation tests
    """
    i: int
    key: str
    latent_dims: list[int]
    loss: list[tuple[float, float]]
    list_dict: list[dict[str, Any]] = []
    mean_losses: list[ndarray] | ndarray = []
    std_losses: list[ndarray] | ndarray = []
    value: dict[str, Any]
    data: dict[str, Any]
    file: BinaryIO

    with open(path, 'rb') as file:
        data = pickle.load(file)

    for value in data.values():
        list_dict.append(value)

    data = {}

    for key in list_dict[0]:
        data[key] = []

    for value in list_dict:
        for key in data:
            data[key].append(value[key])

    data['mean_loss'] = []
    data['std_loss'] = []

    for loss in data['losses']:
        data['mean_loss'].append(np.mean(loss, axis=0))
        data['std_loss'].append(np.std(loss, axis=0))

    for i in range(0, len(data['mean_loss']), 8):
        mean_losses.append(data['mean_loss'][i:i + 8])
        std_losses.append(data['std_loss'][i:i + 8])

    mean_losses = np.array(mean_losses)
    std_losses = np.array(std_losses)
    latent_dims = data['latent_dim'][:8]
    return latent_dims, mean_losses, std_losses


def phys_params(
        data: dict[str, ndarray],
        names: ndarray,
        stellar_frac: ndarray,
        mass: ndarray) -> tuple[list[ndarray], list[ndarray]]:
    """
    Gets several physical parameters for the predicted data and creates a list of the data for each
    simulation

    Parameters
    ----------
    data : dict[str, (M,...)]
        M predicted data
    names : (N) ndarray
        N names for the dataset
    stellar_frac : (N) ndarray
        N stellar fractions for the dataset
    mass : (N) ndarray
        N masses for the dataset

    Returns
    -------
    tuple[list[(B,Z) ndarray], list[(B,6) ndarray]]
        B latent space values of dimensions Z for each simulation and the corresponding B physical
        parameters for each simulation
    """
    value: float
    sim: str
    key: str
    latents: list[ndarray] = []
    params: list[ndarray] = []
    sim_delta_tk: dict[str, float] = {
        'darkskies': 8,
        'hi': 8.2,
        'low': 7.8,
        'bahamas': 8,
        'flamingo': 8.07,
        'tng': 8,
    }
    sim_dm_mass: dict[str, float] = {
        'darkskies': 6.9e7,
        'bahamas': 5.5e9,
        'flamingo': 7.1e8,
        'tng': 6.1e7,
    }
    sim_b_mass: dict[str, float] = {
        'darkskies': 1.1e9,
        'bahamas': 1.1e9,
        'flamingo': 1.3e8,
        'tng': 1.2e7,
    }
    idxs: ndarray
    names = names[data['ids'].astype(int)]
    stellar_frac = stellar_frac[data['ids'].astype(int)]
    mass = mass[data['ids'].astype(int)]

    for sim in names[np.unique(data['targets'], return_index=True)[1]]:
        idxs = sim == names

        for key, value in sim_delta_tk.items():
            if key in sim.lower():
                delta_tk = np.ones(np.count_nonzero(idxs)) * value

        for key, value in sim_dm_mass.items():
            if key in sim.lower():
                dm_mass = np.ones(np.count_nonzero(idxs)) * np.log10(value)

        for key, value in sim_b_mass.items():
            if key in sim.lower():
                b_mass = np.ones(np.count_nonzero(idxs)) * np.log10(value)

        latents.append(data['latent'][idxs])
        params.append(np.stack((
            np.log10(data['targets'][idxs]),
            np.log10(mass[idxs]),
            stellar_frac[idxs],
            delta_tk,
            dm_mass,
            b_mass,
        ), axis=-1))

    return latents, params


def pred_distributions(targets: ndarray, preds: ndarray) -> list[ndarray]:
    """
    Generate the predicted distribution for each simulation

    Parameters
    ----------
    targets : (N) ndarray
        N target values for the predicted data
    preds : (N,...) ndarray
        N predicted data

    Returns
    -------
    list[ndarray]
        Predicted distribution for each simulation
    """
    distributions: list[ndarray] = []

    for target in np.unique(targets):
        distributions.append(preds[targets == target])

    return distributions


def profiles(
        images: ndarray,
        norms: ndarray,
        names: ndarray,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """
    Generates the total, X-ray fraction, and stellar fraction profiles for each simulation

    Parameters
    ----------
    images : (N,3,H,W) ndarray
        N images of height H and width W with channels total mass, X-ray, and stellar
    norms : (N,3) ndarray
        Normalisation factors for each channel
    names : (N) ndarray
        Simulation name for each image

    Returns
    -------
    tuple[(M) ndarray, (L) ndarray, (M,L) ndarray, (M,L) ndarray, (M,L) ndarray]
        M unique simulation names, radii, and total mass, X-ray fraction, and stellar fraction
        profiles with L bins which is equal to int(min(W,H)/4-0.5)
    """
    i: int
    j: int
    radius: int
    name: str
    centers: tuple[int, int]
    idxs: ndarray
    mask: ndarray
    total: ndarray
    x_ray: ndarray
    stellar: ndarray

    images *= norms.reshape(*norms.shape, *[1] * (len(images.shape) - len(norms.shape)))

    centers = (images.shape[-2] // 2, images.shape[-1] // 2)
    total = np.empty((len(np.unique(names)), int(np.ceil((min(centers) - 2) / 2))))
    x_ray = np.empty((len(np.unique(names)), int(np.ceil((min(centers) - 2) / 2))))
    stellar = np.empty((len(np.unique(names)), int(np.ceil((min(centers) - 2) / 2))))

    for i, name in enumerate(np.unique(names)):
        idxs = name == names

        for radius in range(2, min(centers), 2):
            j = (radius - 2) // 2
            mask = np.where(np.sqrt(np.add(*[array ** 2 for array in np.meshgrid(
                np.arange(images.shape[-2]) - images.shape[-2] // 2 + 0.5,
                np.arange(images.shape[-1]) - images.shape[-1] // 2 + 0.5,
            )])) < radius, 1, 0)
            total[i, j] = np.mean(
                np.sum(images[idxs, 0] * mask, axis=(-2, -1)) / (4 * np.pi * radius ** 2),
            )
            x_ray[i, j] = np.mean(
                np.sum(images[idxs, 1] * mask, axis=(-2, -1)) /
                np.sum(images[idxs, 0] * mask, axis=(-2, -1)),
            )
            stellar[i, j] = np.mean(
                np.sum(images[idxs, -1] * mask, axis=(-2, -1)) /
                np.sum(images[idxs, 0] * mask, axis=(-2, -1)),
            )

    return np.unique(names), np.arange(2, min(centers), 2) * 20, total, x_ray, stellar


def summary(data: dict[str, ndarray]) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Generates summary stats for the trained network

    Parameters
    ----------
    data : dictionary
        Data returned from the trained network

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, ndarray]
        Medians, means, standard errors and accuracies for the network predictions
    """
    accuracies: list[float] = []
    stes: list[np.float64] = []
    means: list[np.float64] = []
    medians: list[np.float64] = []
    class_: np.float64
    idxs: ndarray[np.bool_]

    for class_ in np.unique(data['targets']):
        idxs = data['targets'] == class_
        medians.append(np.median(data['latent'][idxs, 0]))
        means.append(np.mean(data['latent'][idxs, 0]))
        stes.append(np.std(data['latent'][idxs, 0]) / np.sqrt(np.count_nonzero(idxs)))
        accuracies.append(np.count_nonzero(data['preds'][idxs] == class_) / len(idxs))

    return np.stack(medians), np.stack(means), np.stack(stes), np.stack(accuracies)
