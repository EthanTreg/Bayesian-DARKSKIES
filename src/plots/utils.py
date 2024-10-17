"""
Utility functions for creating plots
"""
from typing import Sequence

import numpy as np
from numpy import ndarray
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.legend import Legend
from matplotlib.patches import Ellipse
from matplotlib.colors import XKCD_COLORS
from matplotlib.transforms import Transform
from matplotlib.legend_handler import HandlerTuple
from scipy.optimize import minimize

MAJOR: int = 24
MINOR: int = 20
SCATTER_NUM: int = 1000
MARKERS: list[str] = list(Line2D.markers.keys())
COLOURS: list[str] = list(XKCD_COLORS.values())[::-1]
HATCHES: list[str] = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*', '/o', '\\|', '|*', '-\\',
                      '+o', 'x*', 'o-', 'O|', 'O.', '*-']
RECTANGLE: tuple[int, int] = (16, 9)
SQUARE: tuple[int, int] = (10, 10)
HI_RES: tuple[int, int] = (32, 18)
HI_RES_SQUARE: tuple[int, int] = (20, 20)


class UniqueHandlerTuple(HandlerTuple):
    """
    Handler for legend to create handles with unique types per label
    """
    @staticmethod
    def _unique_handles(handles: Sequence[Artist]) -> Sequence[Artist]:
        """
        Returns a list of handles with unique types

        Parameters
        ----------
        handles : Sequence[Artist]
            List of handles

        Returns
        -------
        Sequence[Artist]
            Handles of unique types
        """
        handle: Artist
        return np.array(handles)[np.unique(
            [str(type(handle)) for handle in handles],
            return_index=True,
        )[1]].tolist()

    def create_artists(  # pylint: disable=missing-function-docstring
            self,
            legend: Legend,
            orig_handle: Sequence[Artist],
            xdescent: int,
            ydescent: int,
            width: int,
            height: int,
            fontsize: int,
            trans: Transform) -> Sequence[Artist]:
        artist_list: Sequence[Artist]

        # Generate legend handles and filter unique handles from original handles
        artist_list = self._unique_handles(super().create_artists(
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ))

        # Original handles may be unique, but their legend handles may not, so recreate legend
        # handles with unique handles
        if len(artist_list) != len(orig_handle):
            artist_list = super().create_artists(
                legend,
                artist_list,
                xdescent,
                ydescent,
                width,
                height,
                fontsize,
                trans,
            )
        return artist_list


def contour_sig(counts: float, contour: ndarray) -> float:
    """
    Finds the level that includes the required counts in a contour

    Parameters
    ----------
    counts : float
        Target amount for level to include
    contour : ndarray
        Contour to find the level that gives the target counts

    Returns
    -------
    float
        Level
    """
    return minimize(
        lambda x: np.abs(np.sum(contour[contour > x]) / counts - 1),
        0,
        method='nelder-mead',
    )['x'][0]


def plot_ellipse(
        colour: str,
        data: ndarray,
        axis: Axes,
        stds: list[int] | None = None) -> None:
    """
    Creates confidence ellipse

    Parameters
    ----------
    colour : str
        Colour of the confidence ellipse border
    data : (N,2) ndarray
        N (x,y) data points to generate confidence ellipse for
    axis : Axes
        Axis to add confidence ellipse
    stds : list[int], default = [1]
        The standard deviations of the confidence ellipses
    """
    cov: ndarray
    eig_val: ndarray
    eig_vec: ndarray

    data = data.swapaxes(0, 1)
    cov = np.cov(*data)
    eig_val, eig_vec = np.linalg.eig(cov)
    eig_val = np.sqrt(eig_val)

    if stds is None:
        stds = [1]

    for std in stds:
        axis.add_artist(Ellipse(
            np.mean(data, axis=1),
            width=eig_val[0] * std * 2,
            height=eig_val[1] * std * 2,
            angle=np.rad2deg(np.arctan2(*eig_vec[::-1, 0])),
            facecolor='none',
            edgecolor=colour,
        ))


def subplot_grid(num: int) -> ndarray:
    """
    Calculates the most square grid for a given input for mosaic subplots

    Parameters
    ----------
    num : integer
        Total number to split into a mosaic grid

    Returns
    -------
    ndarray
        2D array of indices with relative width for mosaic subplot
    """
    # Constants
    grid = (int(np.sqrt(num)), int(np.ceil(np.sqrt(num))))
    subplot_layout = np.arange(num)
    diff_row = np.abs(num - np.prod(grid))

    # If number is not divisible into a square-ish grid,
    # then the total number will be unevenly divided across the rows
    if diff_row and diff_row != grid[0]:
        shift_num = diff_row * (grid[1] + np.sign(num - np.prod(grid)))

        # Layout of index and repeated values to correspond to the width of the index
        subplot_layout = np.vstack((
            np.repeat(
                subplot_layout[:-shift_num],
                int(shift_num / diff_row)
            ).reshape(grid[0] - diff_row, -1),
            np.repeat(
                subplot_layout[-shift_num:],
                int((num - shift_num) / (grid[0] - diff_row))
            ).reshape(diff_row, -1),
        ))
    # If a close to square grid is found
    elif diff_row:
        subplot_layout = subplot_layout.reshape(grid[0], grid[1] + 1)
    # If grid is square
    else:
        subplot_layout = subplot_layout.reshape(*grid)

    return subplot_layout
