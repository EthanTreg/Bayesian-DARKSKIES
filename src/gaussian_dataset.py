"""
Generates Gaussian images with different centers
"""
import pickle

import numpy as np


def main():
    """
    Main function to generate Gaussian images
    """
    std = 2e-3
    mu_std = 6e-2
    centers = np.array([0.05, 0.1, 0.3, 0.4, 0.6, 0.7, 0.75, 1])
    shape = [len(centers), int(3e3), 1000, 2]
    image_shape = [2, 100, 100]

    # Initialise date
    mu_offsets = np.random.normal(0, mu_std, size=shape[:2])
    data = np.random.multivariate_normal(
        [0, 0],
        [[std, 0], [0, std]],
        size=np.prod(shape[:3]),
    ).reshape(shape)
    data[..., 0] += centers[:, np.newaxis, np.newaxis] + mu_offsets[..., np.newaxis]
    images = np.zeros((*shape[:2], *image_shape))

    # Convert data x-y values to an image
    data_range = np.stack((np.min(data, axis=(0, 1, 2)), np.max(data, axis=(0, 1, 2))))
    image_data = (
            ((data - data_range[0]) / (data_range[1] + 1e-6 - data_range[0])) * image_shape[1:]
    ).astype(int)
    idxs = np.concatenate((
        np.arange(image_data.shape[0]).repeat(np.prod(image_data.shape[1:3]))[:, np.newaxis],
        np.tile(
            np.arange(image_data.shape[1]).repeat(image_data.shape[-2]),
            image_data.shape[0],
        )[:, np.newaxis],
        image_data.reshape(-1, image_data.shape[-1]),
    ), axis=-1)

    np.add.at(images[:, :, 0], (idxs[:, 0], idxs[:, 1], idxs[:, 2], idxs[:, 3]), 1)
    np.add.at(images[:, :, 1], (idxs[:, 0], idxs[:, 1], idxs[:, 2], idxs[:, 3]), 1)

    # Generate labels and normalise images
    labels = centers.repeat(images.shape[1])
    images = images.reshape(-1, *images.shape[2:])
    images /= np.max(images, axis=(-2, -1))[..., np.newaxis, np.newaxis]

    # Save data
    with open('../data/gaussian_data.pkl', 'wb') as file:
        pickle.dump((labels, images), file)


if __name__ == '__main__':
    main()
