"""
Forward modelling of lensing catalogues.
"""
import os
import pickle
from time import time
from typing import Literal
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy import ndarray
from astropy.io import fits
from astropy import constants as const
from astropy.units import Unit, Quantity
from astropy.cosmology import FLRW, WMAP9  # pylint: disable=no-name-in-module
from scipy.ndimage import map_coordinates
from scipy.stats import gaussian_kde
from netloader.utils import progress_bar

from src.utils.utils import list_dict_convert, ROOT


def sigma_critical(
        z_lens: ndarray | Quantity,
        z_source: ndarray | Quantity,
        cosmology: FLRW) -> Quantity:
    """
    Compute the critical surface mass density for gravitational lensing.

    Parameters
    ----------
    z_lens : ndarray | Quantity
        Lens redshifts with shape (N) and type float
    z_source : ndarray | Quantity
        Source redshifts with shape (N) and type float
    cosmology : FLRW
        An astropy cosmology instance

    Returns
    -------
    Quantity
        Critical surface mass density with shape (N) and type astropy Quantity in units of solar
        masses per square parsec (solMass / pc2)
    """
    factor: float
    z_source: ndarray
    d_ol: ndarray | Quantity
    d_os: ndarray | Quantity
    d_ls: ndarray | Quantity
    result: ndarray | Quantity

    # Ensure vectorization
    z_source = np.atleast_1d(z_source).astype(float)
    assert (z_source >= 0).all(), 'Redshifts must be positive.'

    # Compute distances
    d_ol = cosmology.angular_diameter_distance(z_lens)
    d_os = cosmology.angular_diameter_distance(z_source)
    d_ls = cosmology.angular_diameter_distance_z1z2(z_lens, z_source)

    # Avoid division by zero
    d_ls[d_ls == 0] = np.inf

    # Compute Sigma_crit
    factor = np.power(const.c, 2) / (4 * np.pi * const.G)
    result = factor * d_os / (d_ol * d_ls)

    # Sources at lower z than the halo are not lensed
    result[result <= 0] = np.inf

    if isinstance(result, Quantity):
        return result.to(Unit('solMass / pc2'))
    return result * Unit('solMass / pc2')


def ks93inv(conv_e: ndarray | Quantity, conv_b: ndarray | Quantity) -> ndarray:
    """
    Direct inversion of weak-lensing convergence to shear.

    This function provides the inverse of the Kaiser & Squires (1993) mass
    mapping algorithm, namely the shear is recovered from input E-mode and
    B-mode convergence maps.

    Parameters
    ----------
    conv_e : ndarray | Quantity
        E-mode convergence map with shape (...,H,W) and type float or astropy Quantity where H is
        the height and W is the width of the map
    conv_b : ndarray | Quantity
        B-mode convergence map with shape (...,H,W) and type float or astropy Quantity

    Returns
    -------
    ndarray
        Shear map with shape (...,2,H,W) and type float
    """
    # Check consistency of input maps
    assert conv_e.shape == conv_b.shape

    # Compute Fourier space grids
    k1, k2 = np.meshgrid(np.fft.fftfreq(conv_e.shape[-1]), np.fft.fftfreq(conv_e.shape[-2]))

    # Compute Fourier transforms of kE and kB
    conv_e_hat = np.fft.fft2(conv_e)
    conv_b_hat = np.fft.fft2(conv_b)

    # Apply Fourier space inversion operator
    p1 = k1 * k1 - k2 * k2
    p2 = 2 * k1 * k2
    k2 = k1 * k1 + k2 * k2
    k2[..., 0, 0] = 1  # avoid division by 0
    g1hat = (p1 * conv_e_hat - p2 * conv_b_hat) / k2
    g2hat = (p2 * conv_e_hat + p1 * conv_b_hat) / k2

    # Transform back to pixel space
    g1 = np.fft.ifft2(g1hat).real
    g2 = np.fft.ifft2(g2hat).real
    return np.stack((g1, g2), axis=-3)


def sample_lenses(
        z_lenses: ndarray,
        z_pdf: gaussian_kde,
        size: int = 1000,
        batch: int = 1000) -> tuple[ndarray, ndarray]:
    """
    Sample lens redshifts from a given distribution.

    Parameters
    ----------
    z_lenses : ndarray
        Simulated lens redshifts with shape (N) and type float
    z_pdf : gaussian_kde
        A Gaussian KDE fitted to the observed redshift distribution
    size : int, default = 1000
        The number of samples to draw, M
    batch : int, default = 1000
        The number of samples to draw per iteration

    Returns
    -------
    tuple[ndarray, ndarray]
        Samples with shape (M) and type float and unique redshift indexes with shape (M) and type
        int
    """
    idxs: ndarray
    batch_samples: ndarray
    samples: ndarray = np.array([])
    redshifts: ndarray = np.unique(z_lenses)
    mids: ndarray = np.concat((redshifts[:1], (redshifts[1:] + redshifts[:-1]) / 2, redshifts[-1:]))

    while len(samples) < size:
        batch_samples: ndarray = z_pdf.resample(batch).flatten()
        batch_samples = batch_samples[(mids[0] < batch_samples) & (batch_samples < mids[-1])]
        samples = np.concat((samples, batch_samples))

    samples = samples[:size]
    idxs = np.digitize(samples, mids) - 1
    return samples, idxs


def galaxy_positions(image_shape: tuple[int, int], data: fits.FITS_rec) -> ndarray:
    """
    Extract galaxy positions from a FITS table.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Shape of the output image (height, width)
    data : astropy.io.fits.FITS_rec
        FITS table containing galaxy data with 'ra' and 'dec' columns

    Returns
    -------
    ndarray
        Array of shape (2, N) containing the (x, y) positions of N galaxies
    """
    ra: ndarray = data['ra']
    dec: ndarray = data['dec']
    x: ndarray = (np.mean(ra) - ra) * np.cos(np.mean(dec) * np.pi / 180) * 3600
    y: ndarray = (dec - np.mean(dec)) * 3600

    x = (x - np.min(x)) / (np.max(x) - np.min(x)) * image_shape[0]
    y = (y - np.min(y)) / (np.max(y) - np.min(y)) * image_shape[1]
    return np.stack((x, y), axis=0)


def shear_coords(
        path: str,
        source_dir: str,
        shear_map: ndarray) -> ndarray:
    """
    Extract shear coordinates from a FITS file.

    Parameters
    ----------
    path : str
        Path to the FITS file containing galaxy data
    source_dir : str
        Directory containing source galaxy catalogues
    shear_map : ndarray
        Shear map with shape (2,H,W) and type float where H is the height and W is the width of the
        map

    Returns
    -------
    ndarray
        Array of shape (N,4) and type float containing the positions and shear values (x,y,γ1,γ2)
        of N galaxies
    """
    coords: ndarray
    hdul: fits.HDUList

    with fits.open(os.path.join(source_dir, path)) as hdul:
        coords = galaxy_positions(shear_map.shape[-2:], hdul[1].data)

    return np.stack((
        *coords,
        map_coordinates(shear_map[0], coords),
        map_coordinates(shear_map[1], coords),
    ), axis=1)


def shear_coords_to_image(
        coords: ndarray,
        shear_values: ndarray,
        image_shape: tuple[int, int],
        none_value: float | Literal['mean', 'median'] = 0) -> ndarray:
    """
    Convert shear coordinates to shear images.

    Parameters
    ----------
    coords : ndarray
        Array of shape (N,2) and type float containing the (x,y) positions, where N is the number of
        galaxies
    shear_values : ndarray
        Array of shape (N) and type float containing the shear values
    image_shape : tuple[int, int]
        Shape of the output image (H,W), where H is the height and W is the width of the map
    none_value : float | {'mean', 'median'}, default = 0
        Default value where there are no galaxy samples

    Returns
    -------
    ndarray
        Shear image with shape (H,W) and type float
    """
    image_sum: ndarray
    image_count: ndarray
    none_value = np.mean(shear_values) if none_value == 'mean' else \
    np.median(shear_values) if none_value == 'median' else none_value

    # Convert coordinates to integer pixel indices
    coords = np.clip(np.round(coords).astype(int), 0, np.array(image_shape) - 1)

    # Initialize arrays for sum and count
    image_sum = np.zeros(image_shape)
    image_count = np.zeros(image_shape)

    # Accumulate shear values and counts efficiently
    np.add.at(image_sum, tuple(coords.swapaxes(0, 1)), shear_values)
    np.add.at(image_count, tuple(coords.swapaxes(0, 1)), 1)

    # Compute average, avoiding division by zero
    return np.divide(
        image_sum,
        image_count,
        out=np.ones_like(image_sum) * none_value,
        where=image_count > 0,
    )


def shear_to_image(
        path:str,
        source_dir: str,
        shear_map: ndarray,
        none_value: float | Literal['mean', 'median'] = 0) -> ndarray:
    """
    Convert shear coordinates to shear images.

    Parameters
    ----------
    path : str
        Path to the FITS file containing galaxy data
    source_dir : str
        Directory containing source galaxy catalogues
    shear_map : ndarray
        Shear map with shape (2,H,W) and type float where H is the height and W is the width of the
        map
    none_value : float | {'mean', 'median'}, default = 0
        Default value where there are no galaxy samples

    Returns
    -------
    ndarray
        Shear image with shape (2,H,W) and type float
    """
    coords = shear_coords(path, source_dir, shear_map)
    return np.stack((shear_coords_to_image(
        coords[:, :2],
        coords[:, -2],
        shear_map.shape[-2:],
        none_value=none_value,
    ), shear_coords_to_image(
        coords[:, :2],
        coords[:, -1],
        shear_map.shape[-2:],
        none_value=none_value,
    )))


def main(
        num: int,
        lens_path: str,
        redshifts_path: str,
        output_path: str,
        source_dir: str,
        point: bool = False) -> None:
    """
    Main function to perform forward modelling of lensing catalogues.

    Parameters
    ----------
    num : int
        Number of lensing catalogues to process
    name : str
        Name identifier for the output catalogue
    lens_path : str
        Path to the lens convergence data
    redshifts_path : str
        Path to the file containing observed redshifts
    output_path : str
        Path to save the generated shear catalogues
    source_dir : str
        Directory containing source galaxy catalogues
    point : bool, default = False
        If True, use point catalogues instead of shear maps
    """
    idx: int
    ti: float
    z_lens: float
    z_source: float
    shear_labels: (list[dict[str, float | ndarray]] |
                   dict[str, str | list[float] | list[ndarray] | ndarray]) = []
    shears: list[ndarray] | ndarray = []
    labels: dict[str, str | ndarray]
    images: ndarray
    z_lenses: ndarray
    shear: ndarray
    shear_maps: ndarray
    source_paths: ndarray
    z_lenses_obs: ndarray = np.loadtxt(os.path.join(ROOT, redshifts_path))
    z_lens_pdf: gaussian_kde = gaussian_kde(z_lenses_obs)
    sigma_crit: Quantity

    lens_path = os.path.join(ROOT, lens_path)
    output_path = os.path.join(ROOT, output_path)
    source_dir = os.path.join(ROOT, source_dir)
    source_paths = np.array(os.listdir(source_dir))

    with open(os.path.join(ROOT, lens_path), 'rb') as file:
        labels, images = pickle.load(file)

    print('Generating simulation shear maps...')
    z_source = np.ones(num)
    z_lenses, lens_idxs = sample_lenses(labels['redshift'], z_lens_pdf, size=num)
    sigma_crit = sigma_critical(z_lenses, z_source, WMAP9)
    peaks = labels['norms'][lens_idxs, 0] * Unit('solMass') / (Unit('Mpc') ** 2)
    shear_maps = ks93inv(
        images[lens_idxs, 0] * peaks[:, np.newaxis, np.newaxis] /
        sigma_crit[:, np.newaxis, np.newaxis],
        np.zeros_like(images[lens_idxs, 0]),
    )
    source_paths = np.random.choice(source_paths, size=num)
    ti = time()

    print('Generating shear catalogues...')
    with ProcessPoolExecutor() as executor:
        for i, (idx, z_lens, shear) in enumerate(zip(lens_idxs, z_lenses, executor.map(
            shear_to_image if point else shear_coords,
            source_paths,
            repeat(source_dir),
            shear_maps[lens_idxs],
            *([repeat('median')] if point else [])
        ))):
            shear_labels.append({
                key: value[idx] for key, value in labels.items()
                if isinstance(value, (ndarray, list))
            })
            shear_labels[-1]['z_source'] = z_source
            shear_labels[-1]['z_lens'] = z_lens
            shears.append(shear)
            progress_bar(i, num, text=f'Time: {time() - ti:.1f}s')

    shear_labels = list_dict_convert(shear_labels)

    for key in shear_labels:
        try:
            shear_labels[key] = np.stack(shear_labels[key])
        except ValueError:
            pass

    shear_labels['name'] = f"{labels['name']} FO"

    if point:
        shears = np.stack(shears)
        shear_labels['norms'] = np.max(shears, axis=(1, 2, 3))
        shears /= shear_labels['norms'][:, np.newaxis, np.newaxis, np.newaxis]

    with open(output_path, 'wb') as file:
        pickle.dump((shear_labels, shears), file)


if __name__ == '__main__':
    for sim in ['bahamas_cdm', 'bahamas_0.1', 'bahamas_0.3', 'bahamas_1']:
        main(
            3600,
            f'../data/{sim}.pkl',
            '../data/redshifts.txt',
            f'../data/{sim}_shear_real.pkl',
            '../data/hst_shear/',
            point=True,
        )
