"""
Functions to analyse the neural network
"""
import numpy as np
from numpy import ndarray


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
        Medians, means, standard deviations and accuracies for the network predictions
    """
    accuracies: list[float] = []
    stds: list[np.float64] = []
    means: list[np.float64] = []
    medians: list[np.float64] = []
    class_: np.float64
    idxs: ndarray[bool]

    for class_ in np.unique(data['targets']):
        idxs = data['targets'] == class_
        medians.append(np.median(data['latent'][idxs, 0]))
        means.append(np.mean(data['latent'][idxs, 0]))
        stds.append(np.std(data['latent'][idxs, 0]))
        accuracies.append(np.count_nonzero(data['preds'][idxs] == class_) / len(idxs))

    return np.stack(medians), np.stack(means), np.stack(stds), np.stack(accuracies)
