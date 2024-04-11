"""
Generates Gaussian images with different radii
"""
import pickle

import numpy as np


def main():
    """
    Main function to generate Gaussian images
    """
    num = int(3e3)
    min_counts = 200
    count_density = int(5e3)
    std = 5e-2
    mu_std = 5e-1
    data = []
    ranges = []
    image_shape = [2, 100, 100]
    radii = np.array([0.05, 0.1, 0.3, 0.4, 0.6, 0.7, 0.75, 1])

    # Initialise date
    centers = np.random.normal(0, mu_std, size=[len(radii), num, 1, 2])

    # Generate Gaussian distributions for each radius
    for radius, center in zip(radii, centers):
        counts = max(min_counts, int(count_density * np.pi * radius ** 2))
        data.append(np.random.multivariate_normal(
            [0, 0],
            [[radius, 0], [0, radius]],
            size=num * counts,
        ).reshape([num, counts, 2]))
        radius_scatter = 1 + np.random.normal(0, std, size=(num, 1, 1))
        data[-1] = data[-1] * radius_scatter + center
        ranges.append([np.min(data[-1], axis=(0, 1)), np.max(data[-1], axis=(0, 1))])

    ranges = np.stack(ranges)
    ranges = np.stack([np.min(ranges[:, 0], axis=0), np.max(ranges[:, 1], axis=0)])
    images = np.zeros((len(radii), num, *image_shape))

    # Convert to x-y images for each radii
    for i, datum in enumerate(data):
        image_data = (
                ((datum - ranges[0]) / (ranges[1] + 1e-6 - ranges[0])) * image_shape[1:]
        ).astype(int)
        idxs = np.concatenate((
            np.arange(image_data.shape[0]).repeat(image_data.shape[-2])[:, np.newaxis],
            image_data.reshape(-1, image_data.shape[-1]),
        ), axis=-1)

        np.add.at(images[i, :, 0], tuple([*idxs.swapaxes(0, 1)]), 1)
        np.add.at(images[i, :, 1], tuple([*idxs.swapaxes(0, 1)]), 1)

    # Generate labels and normalise images
    labels = radii.repeat(images.shape[1])
    images = images.reshape(-1, *images.shape[2:])
    images /= np.max(images, axis=(-2, -1))[..., np.newaxis, np.newaxis]

    # Save data
    with open('../data/gaussian_data.pkl', 'wb') as file:
        pickle.dump((labels, images), file)


if __name__ == '__main__':
    main()
