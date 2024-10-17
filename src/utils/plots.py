"""
Creates several plots
"""
import logging
from typing import Any, Callable

import numpy as np
import scienceplots
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.figure import FigureBase
from matplotlib.colors import XKCD_COLORS
from matplotlib.collections import PathCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netloader.utils.utils import label_change
from scipy.stats import gaussian_kde
from numpy import ndarray, floating

from src.plots.utils import subplot_grid, contour_sig

plt.style.use(["science", "grid", 'no-latex'])

MAJOR: int = 24
MINOR: int = 20
SCATTER_NUM: int = 1000
MARKERS: list[str] = list(Line2D.markers.keys())
COLOURS: list[str] = list(XKCD_COLORS.values())[::-1]
HATCHES: list[str] = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
RECTANGLE: tuple[int, int] = (16, 9)
SQUARE: tuple[int, int] = (10, 10)
HI_RES: tuple[int, int] = (32, 18)
HI_RES_SQUARE: tuple[int, int] = (20, 20)


def _init_plot(
        titles: str | list[str] | None = None,
        x_labels: str | list[str] | None = None,
        y_labels: str | list[str] | None = None,
        fig_size: tuple[int, int] = RECTANGLE,
        subfigures: tuple[int, int] | None = None,
        **kwargs: Any) -> tuple[FigureBase, ndarray[FigureBase] | None]:
    """
    Initialises main figure and optionally subfigures

    Parameters
    ----------
    titles : str | list[str], default = None
        Title for the plot or a title for each sub figure
    x_labels : str, default = None
        X label for the plot or x labels for each sub figure
    y_labels : str, default = None
        Y label for the plot or y labels for each sub figure
    fig_size : tuple[int, int]
        Size of the figure
    subfigures : tuple[int, int], default = None
        Number of rows and columns for the sub figures if not None

    **kwargs
        Optional arguments for the subfigures function

    Returns
    -------
    tuple[FigureBase, ndarray[FigureBase] | None]
        FigureBase and subplot axes
    """
    title: str
    x_label: str
    y_label: str
    subfigs: ndarray[FigureBase] | None = None
    subfig: FigureBase
    fig: FigureBase = plt.figure(constrained_layout=True, figsize=fig_size)

    if subfigures:
        subfigs = fig.subfigures(*subfigures, **kwargs)

    if isinstance(titles, list) and subfigs is not None:
        for title, subfig in zip(titles, subfigs.flatten()):
            subfig.suptitle(title, fontsize=MAJOR)
    else:
        fig.suptitle(titles, fontsize=MAJOR)

    if isinstance(x_labels, list) and subfigs is not None:
        for x_label, subfig in zip(x_labels, subfigs.flatten()):
            subfig.supxlabel(x_label, fontsize=MAJOR)
    else:
        fig.supxlabel(x_labels, fontsize=MAJOR)

    if isinstance(y_labels, list) and subfigs is not None:
        for y_label, subfig in zip(y_labels, subfigs.flatten()):
            subfig.supylabel(y_label, fontsize=MAJOR)
    else:
        fig.supylabel(y_labels, fontsize=MAJOR)

    return fig, subfigs


def _init_subplots(
        subplots: str | tuple[int, int] | list | ndarray,
        titles: list[str] | None = None,
        fig_size: tuple[int, int] = RECTANGLE,
        fig: FigureBase | None = None,
        **kwargs: Any) -> tuple[dict[int | str, Axes] | ndarray[Axes], FigureBase]:
    """
    Generates subplots within a figure or sub-figure

    Parameters
    ----------
    subplots : str | tuple[int, int] | list | ndarray
        Parameters for subplots or subplot_mosaic
    titles : list[str], default = None
        Titles for each axis
    fig_size : tuple[int, int]
        Size of the figure, only used if fig is None
    fig : FigureBase | None, default = None
        FigureBase to add subplots to

    **kwargs
        Optional kwargs to pass to subplots or subplot_mosaic

    Returns
    -------
    tuple[dict[int | str, Axes] | ndarray[Axes], FigureBase]
        Dictionary or array of subplot axes and figure
    """
    title: str
    axes: dict[int | str, Axes] | ndarray[Axes]
    axis: Axes

    if fig is None:
        fig = plt.figure(constrained_layout=True, figsize=fig_size)

    # Plots either subplot or mosaic
    if isinstance(subplots, tuple):
        axes = fig.subplots(*subplots, **kwargs)
    else:
        axes = fig.subplot_mosaic(subplots, **kwargs)

    if titles is not None:
        for title, axis in zip(
                titles,
                axes.flatten() if isinstance(axes, ndarray) else axes.values(),
        ):
            axis.set_title(title, fontsize=MAJOR)

    return axes, fig


def _legend(
        labels: ndarray,
        fig: FigureBase,
        columns: int = 2,
        loc: str | tuple[float, float] = 'outside upper center') -> Legend:
    """
    Plots a legend across the top of the plot

    Parameters
    ----------
    labels : (2,L) list | ndarray[np.str_]
        Legend matplotlib handles and labels of size L to be unpacked into handles and labels
    fig : FigureBase
        FigureBase to add legend to
    columns : int, default = 2
        Number of columns for the legend
    loc : str | tuple[float, float], default = 'outside upper center'
        Location to place the legend

    Returns
    -------
    Legend
        Legend object
    """
    rows: int
    fig_size: float = fig.get_size_inches()[0] * fig.dpi
    handle: mpl.artist.Artist
    legend_offset: float
    legend: Legend = fig.legend(
        *labels,
        loc=loc,
        ncol=columns,
        fontsize=MAJOR,
        borderaxespad=0.2,
    )

    legend_offset = float(np.array(legend.get_window_extent())[0, 0])

    if legend_offset < 0:
        legend.remove()
        rows = np.abs(legend_offset) * 2 // fig_size + 2
        columns = np.ceil(len(labels[1]) / rows)
        legend = fig.legend(
            *labels,
            loc=loc,
            ncol=columns,
            fontsize=MAJOR,
            borderaxespad=0.2,
        )

    legend.get_frame().set_boxstyle('Square', pad=0)

    for handle in legend.legend_handles:
        handle.set_alpha(1)

        if isinstance(handle, mpl.collections.PathCollection):
            handle.set_sizes([100])

    return legend


def _legend_marker(
        colours: list[str],
        labels: list[str],
        markers: list[str] | list[None] | ndarray | None = None) -> ndarray:
    """
    Creates markers for a legend

    Parameters
    ----------
    colours : list[string]
        Colours for the legend
    labels : list[string]
        Labels for the legend
    markers : list[string], default = None
        Markers for the legend

    Returns
    -------
    ndarray
        Legend labels
    """
    label: str
    colour: str
    marker: str | None
    legend_labels: list[tuple[PathCollection, str]] = []

    if markers is None:
        markers = [None] * len(colours)

    for colour, label, marker in zip(colours, labels, markers):
        legend_labels.append((plt.gca().scatter([], [], color=colour, marker=marker), label))

    return np.array(legend_labels).swapaxes(0, 1)


def _plot_clusters_2d(
        label: str,
        colour: str,
        data: ndarray[floating],
        ranges: ndarray[floating],
        axes: ndarray[Axes],
        density: bool = True,
        bin_num: int = 200,
        hatch: str | None = None,
        markers: ndarray[np.str_] | None = None) -> None:
    """
    Plots scattered data with contours and histograms to show data distribution for 2D data

    Parameters
    ----------
    colour : str
        Colour of the data
    label : str
        Label for the data
    data : (N,2) ndarray[floating]
        N (x,y) data points to plot
    ranges : (2,2) ndarray[floating]
        Min and max values for the x and y axes
    axes : (2,2) ndarray[Axes]
        Axes to plot histograms and the scatter plot
    density : bool, default = True
        If density contours should be plotted or confidence ellipses
    bin_num : int, default = 200
        Resolution of the density plot contours or number of histogram bin_num
    hatch : str, default = None
        Hatching pattern for the contour
    markers : ndarray[str_], default = None
        Markers for the data
    """
    scat_alpha: float = 0.2
    hist_alpha: float = 0.4
    marker: str
    marker_idxs: ndarray[np.bool_]

    _plot_histogram(
        data[:, 0],
        axes[0, 0],
        density=density,
        bin_num=bin_num,
        alpha=hist_alpha,
        colour=colour,
        data_range=ranges[0],
    )
    axes[1, 0].tick_params(labelsize=MINOR)
    axes[1, 0].set_xlim(ranges[0])
    axes[1, 0].set_ylim(ranges[1])
    _plot_histogram(
        data[:, 1],
        axes[1, 1],
        density=density,
        bin_num=bin_num,
        alpha=hist_alpha,
        colour=colour,
        data_range=ranges[1],
        orientation='horizontal',
    )

    if markers is not None:
        for marker in np.unique(markers):
            marker_idxs = marker == markers
            axes[1, 0].scatter(
                data[marker_idxs, 0],
                data[marker_idxs, 1],
                alpha=scat_alpha,
                color=colour,
                label=label,
                marker=marker,
            )
    else:
        axes[1, 0].scatter(data[:, 0], data[:, 1], alpha=scat_alpha, color=colour, label=label)

    if density:
        _plot_density(colour, data, ranges, axes[1, 0], bin_num=bin_num, hatch=hatch)
    else:
        _plot_ellipse(colour, data, axes[1, 0], stds=[1, 2])


def _plot_clusters_3d(
        label: str,
        colour: str,
        data: ndarray[floating],
        ranges: ndarray[floating],
        axis: Axes,
        bin_num: int = 200,
        hatch: str | None = None,
        markers: ndarray[np.str_] | None = None) -> None:
    """
    Plots scattered data with contours and histograms to show data distribution for 2D data

    Parameters
    ----------
    label : str
        Label for the data
    colour : str
        Colour of the data
    data : Nx3 ndarray[floating]
        N (x,y,z) data points to plot
    axis : Axes
        Axis to plot 3D scatter plot with contours
    ranges : 2x2 ndarray[floating]
        Min and max values for the x and y axes
    bin_num : int, default = 200
        Resolution of the density plot contours
    hatch : str, default = None
        Hatching pattern for the contour
    markers : ndarray[str_], default = None
        Markers for the data
    """
    alpha: float = 0.4
    marker: str
    order: list[int]
    axes: list[str] = ['x', 'y', 'z']
    orders: list[list[int]] = [[0, 1, 2], [0, 2, 1], [2, 1, 0]]
    marker_idxs: ndarray[np.bool_]

    if markers is not None:
        for marker in np.unique(markers):
            marker_idxs = marker == markers
            axis.scatter(
                *data[marker_idxs].swapaxes(0, 1),
                alpha=alpha,
                color=colour,
                label=label,
                marker=marker,
            )
    else:
        axis.scatter(*data.swapaxes(0, 1), alpha=alpha, color=colour, label=label)

    for order in orders:
        _plot_density(
            colour,
            ranges[order[:2]],
            data[:, order[:2]],
            axis,
            bin_num=bin_num,
            alpha=alpha,
            hatch=hatch,
            order=order,
            zdir=axes[order[-1]],
            offset=ranges[order[-1], 0],
        )


def _plot_density(
        colour: str,
        data: ndarray[floating],
        ranges: ndarray[floating],
        axis: Axes,
        bin_num: int = 200,
        alpha: float = 0.2,
        hatch: str | None = None,
        order: list[int] | None = None,
        confidences: list[float] | None = None,
        **kwargs: Any) -> None:
    """
    Plots a density contour plot

    Parameters
    ---------
    colour : str
        Colour of the contour
    data : (N,2) ndarray[floating]
        N (x,y) data points to generate density contour for
    ranges : (2,2) ndarray[floating]
        Min and max values for the x and y axes
    axis : Axes
        Axis to add density contour
    bin_num : int, default = 200
        Resolution of the contours
    alpha : float, default = 0.2
        Alpha value for the contour
    hatch : str, default = None
        Hatching pattern for the contour
    confidences : list[float], default = [0.68]
        List of confidence values to plot contours for, starting with the lowest confidence

    **kwargs
        Optional kwargs to pass to Axes.contour and Axes.contourf
    """
    total: float
    levels: list[float]
    logger: logging.Logger = logging.getLogger(__name__)
    contour: ndarray
    grid: ndarray = np.mgrid[
                    ranges[0, 0]:ranges[0, 1]:bin_num * 1j,
                    ranges[1, 0]:ranges[1, 1]:bin_num * 1j,
                    ]
    kernel: gaussian_kde

    try:
        kernel = gaussian_kde(data.swapaxes(0, 1))
    except np.linalg.LinAlgError:
        logger.warning('Cannot calculate contours, skipping')
        return

    contour = np.reshape(kernel(grid.reshape(2, -1)).T, (bin_num, bin_num))
    total = np.sum(contour)
    levels = [np.max(contour)]

    if confidences is None:
        confidences = [0.68]

    for confidence in confidences:
        levels.insert(0, contour_sig(total * confidence, contour))

    if levels[-1] == 0:
        logger.warning('Cannot calculate contours, skipping')
        return

    contour = np.concatenate((grid, contour[np.newaxis]), axis=0)

    if order is not None:
        contour = contour[order]

    axis.contourf(*contour, levels, alpha=alpha, colors=colour, hatches=[hatch], **kwargs)
    axis.contour(*contour, levels, colors=colour, **kwargs)


def _plot_ellipse(
        colour: str,
        data: ndarray[floating],
        axis: Axes,
        stds: list[int] | None = None) -> None:
    """
    Creates confidence ellipse

    Parameters
    ----------
    colour : str
        Colour of the confidence ellipse border
    data : Nx2 ndarray
        N (x,y) data points to generate confidence ellipse for
    axis : Axes
        Axis to add confidence ellipse
    stds : list[int], default = [1]
        The standard deviations of the confidence ellipses
    """
    cov: ndarray
    eig_val: ndarray[floating]
    eig_vec: ndarray[floating]

    data = data.swapaxes(0, 1)
    cov = np.cov(*data)
    eig_val, eig_vec = np.linalg.eig(cov)
    eig_val = np.sqrt(eig_val)

    if stds is None:
        stds = [1]

    for std in stds:
        axis.add_artist(mpl.patches.Ellipse(
            np.mean(data, axis=1),
            width=eig_val[0] * std * 2,
            height=eig_val[1] * std * 2,
            angle=np.rad2deg(np.arctan2(*eig_vec[::-1, 0])),
            facecolor='none',
            edgecolor=colour,
        ))


def _plot_histogram(
        data: ndarray[floating],
        axis: Axes,
        log: bool = False,
        density: bool = False,
        bin_num: int = 100,
        alpha: float | None = None,
        colour: str = COLOURS[0],
        hatch: str | None = None,
        labels: str | list[str] | list[None] | None = None,
        data_range: tuple[float, float] | None = None,
        data_twin: ndarray[np.floating] = None,
        **kwargs: Any) -> None | Axes:
    """
    Plots a histogram subplot with twin data if provided

    Parameters
    ----------
    data : ndarray[floating]
        Primary data to plot
    axis : Axes
        Axis to plot on
    log : bool, default = False
        If data should be plotted on a log scale, expects linear data
    density : bool, default = False
        If histogram should be plotted as a kernel density estimation
    bin_num : int, default = 100
        Number of bin_num
    alpha : float, default = 0.2 if density, 0.5 if data_twin is provided; otherwise, 1
        Transparency of the histogram, gets halved if data_twin is provided
    colour : str, default = COLOURS[0]
        Colour of the histogram or density plot
    labels : list[str], default = None
        Labels for data and, if provided, data_twin
    data_range : tuple[float, float], default = None
        x-axis data range, required if density is True
    data_twin : ndarray[floating], default = None
        Secondary data to plot

    **kwargs
        Optional keyword arguments that get passed to Axes.plot and Axes.fill_between if density is
        True, else to Axes.hist

    Returns
    -------
    None | Axes
        If data_twin is provided, returns twin axis
    """
    bins: int | ndarray = bin_num
    x_data: ndarray
    y_data: ndarray
    twin_axis: Axes
    kernel: gaussian_kde

    if not isinstance(labels, list):
        labels = [labels] * 2

    if alpha is None and density:
        alpha = 0.2
    elif alpha is None and data_twin is None:
        alpha = 1
    else:
        alpha = 0.5

    if log:
        bins = np.logspace(*np.log10(data_range), bin_num)
        axis.set_xscale('log')

    if density and len(np.unique(data)) > 1:
        kernel = gaussian_kde(data)
        x_data = np.linspace(*data_range, bin_num)
        y_data = kernel(x_data)

        if 'orientation' in kwargs and kwargs['orientation'] == 'horizontal':
            del kwargs['orientation']
            axis.plot(y_data, x_data, label=labels[0], color=colour, **kwargs)
            axis.fill_betweenx(x_data, y_data, alpha=alpha, color=colour, hatch=hatch, **kwargs)
        else:
            axis.plot(x_data, y_data, label=labels[0], color=colour, **kwargs)
            axis.fill_between(x_data, y_data, alpha=alpha, color=colour, hatch=hatch, **kwargs)
    else:
        axis.hist(
            data,
            bins=bins,
            alpha=alpha,
            label=labels[0],
            color=colour,
            range=data_range,
            **kwargs,
        )

    axis.tick_params(labelsize=MINOR)
    axis.ticklabel_format(axis='y', scilimits=(-2, 2))

    if data_twin is not None:
        twin_axis = axis.twinx()
        _plot_histogram(
            data_twin,
            twin_axis,
            log=log,
            density=density,
            bin_num=bin_num,
            alpha=alpha,
            colour=COLOURS[1],
            hatch=hatch,
            labels=[labels[1]],
            data_range=data_range,
            **kwargs,
        )
        return twin_axis
    return None


def _plot_histogram2(
        data: ndarray[floating],
        axis: Axes,
        log: bool = False,
        density: bool = False,
        bin_num: int = 100,
        alpha: float | None = None,
        colour: str = COLOURS[0],
        hatch: str | None = None,
        data_range: tuple[float, float] | None = None,
        **kwargs: Any) -> None:
    """
    Plots a histogram subplot with twin data if provided

    Parameters
    ----------
    data : ndarray[floating]
        Primary data to plot
    axis : Axes
        Axis to plot on
    log : bool, default = False
        If data should be plotted on a log scale, expects linear data
    density : bool, default = False
        If histogram should be plotted as a kernel density estimation
    bin_num : int, default = 100
        Number of bin_num
    alpha : float, default = 0.2 if density, 0.5 if data_twin is provided; otherwise, 1
        Transparency of the histogram, gets halved if data_twin is provided
    colour : str, default = COLOURS[0]
        Colour of the histogram or density plot
    labels : list[str], default = None
        Labels for data and, if provided, data_twin
    data_range : tuple[float, float], default = None
        x-axis data range, required if density is True
    data_twin : ndarray[floating], default = None
        Secondary data to plot

    **kwargs
        Optional keyword arguments that get passed to Axes.plot and Axes.fill_between if density is
        True, else to Axes.hist

    Returns
    -------
    None | Axes
        If data_twin is provided, returns twin axis
    """
    bins: int | ndarray = bin_num
    x_data: ndarray
    y_data: ndarray
    kernel: gaussian_kde

    if alpha is None and density:
        alpha = 0.2
    else:
        alpha = 0.5

    if log:
        bins = np.logspace(*np.log10(data_range), bin_num)
        axis.set_xscale('log')

    if density and len(np.unique(data)) > 1:
        kernel = gaussian_kde(data)
        x_data = np.linspace(*data_range, bin_num)
        y_data = kernel(x_data)

        if 'orientation' in kwargs and kwargs['orientation'] == 'horizontal':
            del kwargs['orientation']
            axis.plot(y_data, x_data, label=labels[0], color=colour, **kwargs)
            axis.fill_betweenx(x_data, y_data, alpha=alpha, color=colour, hatch=hatch, **kwargs)
        else:
            axis.plot(x_data, y_data, label=labels[0], color=colour, **kwargs)
            axis.fill_between(x_data, y_data, alpha=alpha, color=colour, hatch=hatch, **kwargs)
    else:
        y_data, x_data = np.histogram(data, bins)
        y_data = y_data / np.max(y_data)
        axis.bar(
            x_data[:-1],
            y_data,
            align='edge',
            width=np.diff(x_data),
            alpha=alpha,
            color=colour,
            **kwargs,
        )

    axis.tick_params(labelsize=MINOR)
    axis.ticklabel_format(axis='y', scilimits=(-2, 2))


def _plot_reconstruction(
        x_data: list[ndarray] | ndarray,
        targets: list[ndarray] | ndarray,
        preds: list[ndarray] | ndarray,
        axis: Axes,
        labels: str | list[str] | list[None] | None = None,
        colours: str | list[str] | list[None] | None = None,
        uncertainties: list[ndarray] | list[None] | ndarray | None = None) -> Axes:
    """
    Plots reconstructions and residuals

    Parameters
    ----------
    x_data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
        x-axis values, where N are the number of points per run and B are the number of plots
    targets : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
        Target y values, where N are the number of points per run and B are the number of plots
    preds : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
        Predicted y values, where N are the number of points per run and B are the number of plots
    axis : Axes
        Plot axes
    labels : str | list[str], default = None
        Legend label for each input data
    colours : str | list[str], default = COLOURS
        Colours to use for plotting
    uncertainties : list[(N) ndarray] | (N) ndarray | (B,N) ndarray, default = None
        Uncertainties in the y predictions, where N are the number of points per run and B are the
        number of plots

    Returns
    -------
    Axes
        Reconstruction axis
    """
    if isinstance(preds, ndarray) and len(preds.shape) == 1:
        preds = [preds]

    if not isinstance(labels, list):
        labels = [labels] * len(preds)

    if colours is None:
        colours = COLOURS
    elif isinstance(colours, str):
        colours = [colours] * len(preds)

    if isinstance(x_data, ndarray) and len(x_data.shape) == 1:
        x_data = [x_data] * len(preds)

    if isinstance(targets, ndarray) and len(targets.shape) == 1:
        targets = [targets] * len(preds)

    if not isinstance(uncertainties, list) and len(uncertainties.shape) == 1:
        uncertainties = [uncertainties] * len(preds)

    major_axis: Axes = make_axes_locatable(axis).append_axes('top', size='150%', pad=0)

    for label, colour, x_datum, target, pred, uncertainty in zip(
            labels,
            colours,
            x_data,
            targets,
            preds,
            uncertainties,
    ):
        axis.scatter(x_datum, pred - target, marker='x', color=colour)

        major_axis.scatter(x_datum, pred, label=label, color=colour, marker='x')

        if not (x_datum == target).all():
            major_axis.scatter(x_datum, target, label=f'{label} Target')

        if uncertainty is not None:
            major_axis.fill_between(
                x_datum,
                pred - uncertainty,
                pred + uncertainty,
                color=colour,
                alpha=0.1,
            )
            axis.fill_between(
                x_datum,
                pred - uncertainty - target,
                pred + uncertainty - target,
                color=colour,
                alpha=0.1,
            )

    # axis.locator_params(axis='y', nbins=4)
    axis.tick_params(axis='both', labelsize=MINOR)
    axis.set_ylabel('Residual', fontsize=MAJOR)
    axis.hlines(0, xmin=np.min(x_data), xmax=np.max(x_data), color='k')
    # major_axis.locator_params(axis='y', nbins=4)
    major_axis.tick_params(axis='y', labelsize=MINOR)
    major_axis.set_xticks([])
    return major_axis


def plot_clusters(
        path: str,
        data: ndarray[floating],
        classes: ndarray[floating],
        density: bool = True,
        plot_3d: bool = False,
        bin_num: int = 200,
        colours: list[str] | None = None,
        hatches: list[str] | list[None] | None = None,
        labels: list[str] | list[None] | None = None,
        predictions: ndarray[floating] | None = None) -> None:
    """
    Plots clusters either as a 2D scatter plot or 1D histogram

    Parameters
    ----------
    path : str
        Path to save plots
    data : NxD ndarray[floating]
        N cluster data points of dimension D = {1,2}
    classes : N ndarray[floating]
        Data classes for N data points
    density : bool, default = True
        If density contours should be plotted or confidence ellipses
    plot_3d : bool, default = False
        If 3D plot or corner plot should be used for 3D data
    bin_num : int, default = 200
        Resolution of the density plot contours or number of bin_num if density is False
    colours : list[str], default = COLOURS
        Class colours
    hatches : list[str], default = None
        Hatching pattern for the contours
    labels : list[str], default = None
        Class labels
    predictions : N ndarray[floating], default = None
        Class predicted labels
    """
    class_: float
    pad: float = 0.05
    colour: str
    label: str | None
    label_idxs: ndarray[np.bool_]
    class_data: ndarray[floating]
    ranges: ndarray[floating] = np.stack((
        np.min(data - np.abs(pad * data), axis=0),
        np.max(data + np.abs(pad * data), axis=0),
    ), axis=1)
    markers: ndarray[np.str_] | None = None
    axes: ndarray[Axes]
    legend_labels: ndarray
    fig: FigureBase
    axis: Axes
    colours = COLOURS if colours is None else colours

    if labels is None:
        labels = [None] * np.unique(classes).size

    if hatches is None:
        hatches = [None] * np.unique(classes).size

    if data.shape[1] == 1:
        fig = plt.figure(figsize=RECTANGLE, constrained_layout=True)
    elif data.shape[1] == 2:
        axes, fig = _init_subplots(
            (2, 2),
            sharex='col',
            sharey='row',
            width_ratios=[3, 1],
            height_ratios=[1, 3],
        )
        axes[0, 1].remove()
        axes[0, 0].tick_params(bottom=False)
        axes[1, 1].tick_params(left=False)
    elif data.shape[1] == 3 and plot_3d:
        fig = plt.figure(figsize=SQUARE, constrained_layout=True)
        axis = fig.add_subplot(projection='3d')
        axis.set_xlim(ranges[0])
        axis.set_ylim(ranges[1])
        axis.set_zlim(ranges[2])
        axis.invert_yaxis()
    else:
        axes, fig = _init_subplots((data.shape[1],) * 2, fig_size=HI_RES, sharex='col')

    if labels[0] is not None:
        legend_labels = _legend_marker(colours, labels, None if predictions is None else MARKERS)
    else:
        legend_labels = _legend_marker(colours, labels)

    # Plot each cluster class
    for class_, colour, hatch, label in zip(np.unique(classes), colours, hatches, labels):
        label_idxs = classes == class_
        class_data = data[label_idxs]

        if predictions is not None:
            markers = np.array([MARKERS[0]] * len(class_data))

            for class_j, marker in zip(np.unique(classes), MARKERS):
                markers[predictions[label_idxs] == class_j] = marker

        # Plot data depending on number of dimensions
        if data.shape[1] == 1:
            _plot_histogram(
                class_data,
                plt.gca(),
                alpha=0.4,
                colour=colour,
                labels=label,
                data_range=(np.min(data), np.max(data)),
                hatch=hatch,
            )
        elif data.shape[1] == 2:
            _plot_clusters_2d(
                label,
                colour,
                class_data,
                ranges,
                axes,
                density=density,
                bin_num=bin_num,
                hatch=hatch,
                markers=markers,
            )
        elif data.shape[1] == 3 and plot_3d:
            _plot_clusters_3d(
                label,
                colour,
                class_data,
                ranges,
                axis,
                bin_num=bin_num,
                hatch=hatch,
                markers=markers,
            )
        else:
            plot_param_pairs(
                class_data,
                density=density,
                ranges=ranges,
                axes=axes,
                bin_num=bin_num,
                colour=colour,
                hatch=hatch,
            )

    if data.shape[1] > 1 and labels[0] is not None:
        _legend(legend_labels, fig, columns=len(labels))

    plt.savefig(f'{path}.png')


def plot_confusion(
        plots_dir: str,
        labels: list[str],
        targets: ndarray[floating],
        predictions: ndarray[floating]) -> None:
    """
    Plots the confusion matrix between targets and predictions

    Parameters
    ----------
    plots_dir : str
        Directory to save plots
    labels : list[str]
        Labels for each class
    targets : N ndarray[floating]
        Target values
    predictions : N ndarray[floating]
        Predicted values
    """
    i: int
    j: int
    value: float
    class_: float
    colour: str
    idxs: ndarray[np.bool_]
    counts: ndarray[np.int_]
    matrix_row: ndarray[floating]
    class_preds: ndarray[floating]
    classes: ndarray[floating] = np.unique(targets)
    matrix: ndarray[floating] = np.zeros((len(classes), len(classes)))
    plt.figure(figsize=SQUARE if len(classes) < 5 else HI_RES_SQUARE, constrained_layout=True)

    # Generate confusion matrix
    for matrix_row, class_ in zip(matrix, classes):
        idxs = targets == class_
        class_preds, counts = np.unique(predictions[idxs], return_counts=True)
        class_preds = label_change(class_preds, classes)
        matrix_row[class_preds] = counts / np.count_nonzero(idxs) * 100

    plt.imshow(matrix, cmap='Blues')
    plt.xticks(np.arange(len(labels)), labels, fontsize=MINOR)
    plt.yticks(np.arange(len(labels)), labels, rotation=90, va='center', fontsize=MINOR)
    plt.xlabel('Predictions', fontsize=MAJOR)
    plt.ylabel('Targets', fontsize=MAJOR)

    for (i, j), value in zip(np.ndindex(matrix.shape), matrix.flatten()):
        colour = 'w' if value > 50 else 'k'
        plt.text(j, i, f'{value:.1f}', ha='center', va='center', color=colour, fontsize=MINOR)

    plt.savefig(f'{plots_dir}Confusion_Matrix.png')


def plot_distributions(
        plots_dir: str,
        name: str,
        data: list[ndarray[floating]] | ndarray[floating],
        log: bool = False,
        y_axis: bool = True,
        num_plots: int = 12,
        x_label: str | None = None,
        text_loc: str = 'upper right',
        titles: list[str] | None = None,
        labels: list[str] | list[None] | None = None,
        texts: list[str] | list[None] | None = None,
        data_twin: list[ndarray[floating] | None] | ndarray[floating] | None = None,
        axes: dict[int | str, Axes] | None = None,
        **kwargs: Any) -> dict[int | str, Axes]:
    """
    Plots the distributions for a number of examples

    Parameters
    ----------
    plots_dir : str
        Directory to save plots
    name : str
        File name to save plot
    data : list[ndarray[floating]] | ndarray[floating]
        Distributions to plot, each row is a different distribution
    log : bool, default = False,
        If the x-axis should be logged
    y_axis : bool, default = True
        If y-axis should be plotted
    num_plots : int, default = 16
        Number of distributions to plot, number of rows in data will be used if rows < num_plots
    x_label : str, default = None
        Label of the x-axis
    text_loc : str, default = 'upper right'
        Location of the text
    labels : list[str], default = None
        Labels for data and data_twin if provided
    titles : list[str], default = None
        Titles for the distributions
    texts : list[str], default = None
        Text to display on each plot
    data_twin : list[ndarray[floating]], ndarray[floating], default = None
        Twin distributions to plot, each row is a different distribution corresponding to data
    axes : dict[int | str, Axes], default = None
        Dictionary of axes for each subplot

    **kwargs
        Optional keyword arguments for plotting the histogram

    Returns
    -------
    dict[int | str, Axes]
        Dictionary of axes for each subplot
    """
    datum: ndarray[floating]
    datum_twin: ndarray[floating] | None
    fig: FigureBase
    axis: Axes
    twin_axis: Axes | None = None

    if axes is None:
        axes, fig = _init_subplots(
            subplot_grid(min(len(data), num_plots)),
            titles=titles,
            fig=_init_plot(x_labels=x_label)[0],
        )
    else:
        fig = axes[0].get_figure()

    if labels is None:
        labels = [None, None]

    if data_twin is None:
        data_twin = [None] * len(data)

    if texts is None:
        texts = [None] * len(data)

    for datum, datum_twin, text, axis in zip(data, data_twin, texts, axes.values()):
        axis.set_xscale('log' if log else 'linear')
        twin_axis = _plot_histogram(
            datum,
            axis,
            log=log,
            labels=labels,
            data_twin=datum_twin,
            **kwargs,
        )
        # twin_axis = _plot_histogram2(
        #     datum_twin,
        #     axis,
        #     log=log,
        #     colour=COLOURS[1],
        #     **kwargs,
        # )

        if not y_axis:
            axis.tick_params(labelleft=False, left=False)

        if twin_axis and not y_axis:
            twin_axis.tick_params(labelright=False, right=False)

        if text:
            axis.add_artist(mpl.offsetbox.AnchoredText(
                text,
                loc=text_loc,
                prop={'fontsize': MINOR},
            ))

    if labels[0] is not None and twin_axis is not None:
        _legend(np.hstack((
            axes[0].get_legend_handles_labels(),
            twin_axis.get_legend_handles_labels(),
        )), fig)
    elif labels[0] is not None:
        _legend(axes[0].get_legend_handles_labels(), fig)

    if plots_dir and name:
        plt.savefig(f'{plots_dir}{name}.png')

    return axes


def plot_gaussian_preds(
        plots_dir: str,
        data: dict[str, list[str] | list[list[float]] | list[ndarray]]) -> None:
    """
    Plots the Gaussian test predictions

    Parameters
    ----------
    plots_dir : str
        Directory to save plots
    data : dict[str, list[str] | list[list[float]] | list[ndarray]]
        Data return from gaussian_tests
    """
    major_axis: Axes

    fig = plt.figure(constrained_layout=True, figsize=RECTANGLE)
    major_axis = _plot_reconstruction(
        np.array(data['unseen']),
        np.array(data['unseen']),
        np.mean(data['means'], axis=1),
        plt.gca(),
        labels=data['names'],
        colours=COLOURS,
        uncertainties=np.std(data['means'], axis=1),
    )
    major_axis.plot(
        [np.min(data['unseen']), np.max(data['unseen'])],
        [np.min(data['unseen']), np.max(data['unseen'])],
        color='k',
        label='Target',
    )
    _legend(major_axis.get_legend_handles_labels(), fig)
    plt.savefig(f'{plots_dir}Gaussian_Predictions.png')


def plot_multi_plot(
        labels: list[str],
        data: list[ndarray],
        plot_func: Callable[..., dict[int | str, Axes] | ndarray],
        name: str = '',
        plots_dir: str = '',
        **kwargs: Any) -> None:
    """
    Plots multiple datasets onto the same plot

    Parameters
    ----------
    labels : list[str]
        Labels for the different datasets
    data : list[(N,L) ndarray]
        Datasets to plot
    plot_func : Callable[..., dict[int | str, Axes] | ndarray]
        Plotting function to plot multiple datasets onto
    name : str, default = ''
        Name of the plot
    plots_dir : str, default = ''
        Directory to save plots
    """
    datum: ndarray
    axes: dict[int | str, Axes] | ndarray | None = None

    for colour, datum in zip(COLOURS, data):
        axes = plot_func(datum, axes=axes, alpha=len(data) ** -1, colour=colour, **kwargs)

    _legend(_legend_marker(COLOURS, labels), plt.gcf(), columns=len(labels))

    if plots_dir and name:
        plt.savefig(f'{plots_dir}{name}.png')


def plot_param_comparison(
        plots_dir: str,
        x_data: ndarray[floating],
        y_data: ndarray[floating]) -> None:
    """
    Plots y_data against x_data for comparison

    Parameters:
    ----------
    plots_dir : str
        Directory to save plots
    x_data : ndarray[floating]
        X-axis data
    y_data : ndarray[floating]
        Y-axis data
    """
    value_range: tuple[float, float] = (np.min(x_data), np.max(x_data))
    axis: Axes

    plt.figure(figsize=RECTANGLE, constrained_layout=True)
    axis = plt.gca()

    axis.scatter(x_data, y_data, alpha=0.2)
    axis.plot(value_range, value_range, color='k')
    axis.tick_params(labelsize=MINOR)
    axis.xaxis.get_offset_text().set_visible(False)
    axis.yaxis.get_offset_text().set_size(MINOR)
    axis.text(
        0.1,
        0.9,
        rf'$\chi^2_\nu=${np.mean((y_data - x_data) ** 2 / x_data):.2f}',
        fontsize=MINOR,
        transform=axis.transAxes,
    )
    plt.savefig(f'{plots_dir}Parameter_Comparison.png')


def plot_param_pairs(
        data: ndarray[floating],
        density: bool = False,
        plots_dir: str | None = None,
        ranges: ndarray[floating] | None = None,
        axes: ndarray[Axes] | None = None,
        **kwargs: Any) -> None:
    """
    Plots a pair plot to compare the distributions and comparisons between parameters

    Parameters
    ----------
    data : (N,L) ndarray[floating]
        Data to plot parameter pairs for N data points and L parameters
    density : bool, default = False
        If density contours should be plotted
    plots_dir : str, default = None
        Directory to save plots
    ranges : (L,2) ndarray[floating], default = None
        Ranges for L parameters, required if using kwargs to plot densities
    axes : (L,L) ndarray[Axes], default = None
        Axes to use for plotting L parameters

    **kwargs
        Optional keyword arguments passed to _plot_histogram and _plot_density
    """
    i: int
    j: int
    x_data: ndarray[floating]
    y_data: ndarray[floating]
    x_range: ndarray[floating]
    y_range: ndarray[floating]
    axes_row: ndarray[Axes]
    axis: Axes
    data = np.swapaxes(data, 0, 1)

    if ranges is None:
        ranges = [None] * data.shape[0]

    if axes is None:
        axes, _ = _init_subplots((data.shape[0],) * 2, sharex='col')

    # Loop through each subplot
    for i, (axes_row, y_data, y_range) in enumerate(zip(axes, data, ranges)):
        for j, (axis, x_data, x_range) in enumerate(zip(axes_row, data, ranges)):
            # Share y-axis for all scatter plots
            if i != j:
                axis.sharey(axes_row[0])

            # Set number of ticks
            axis.locator_params(axis='x', nbins=3)
            axis.locator_params(axis='y', nbins=3)

            # Hide ticks for plots that aren't in the first column or bottom row
            if j == 0:
                axis.tick_params(axis='y', labelsize=MINOR)
            else:
                axis.tick_params(labelleft=False, left=False)

            if i == axes.shape[0] - 1:
                axis.tick_params(axis='x', labelsize=MINOR)
            else:
                axis.tick_params(labelbottom=False, bottom=False)

            if x_range is not None and j < i:
                axis.set_xlim(x_range)
                axis.set_ylim(y_range)

            # Plot scatter plots & histograms
            if i == j:
                _plot_histogram(x_data, axis, density=density, data_range=x_range, **kwargs)
                axis.tick_params(labelleft=False, left=False)
            elif j < i:
                axis.scatter(x_data[:SCATTER_NUM], y_data[:SCATTER_NUM], s=4, alpha=0.3)

                if density:
                    _plot_density(
                        ranges=np.array((x_range, y_range)),
                        data=np.stack((x_data, y_data), axis=1),
                        axis=axis,
                        **kwargs,
                    )
            else:
                axis.set_visible(False)

    if plots_dir:
        plt.savefig(f'{plots_dir}Parameter_Pairs.png')


def plot_performance(
        plots_dir: str,
        name: str,
        y_label: str,
        val: list[float] | ndarray[floating],
        log_y: bool = True,
        train: list[float] | ndarray[floating] | None = None) -> None:
    """
    Plots training and validation performance as a function of epochs

    Parameters
    ----------
    plots_dir : str
        Directory to save plots
    name : str
        File name to save plot
    y_label : str
        Performance metric
    val : list[float] | ndarray[floating]
        Validation performance
    log_y : bool, default = True
        If y-axis should be logged
    train : list[float] | ndarray[floating], default = None
        Training performance
    """
    plt.figure(figsize=RECTANGLE, constrained_layout=True)

    if train is not None:
        plt.plot(train, label='Training')

    plt.plot(val, label='Validation')
    plt.xticks(fontsize=MINOR)
    plt.yticks(fontsize=MINOR)
    plt.yticks(fontsize=MINOR, minor=True)
    plt.xlabel('Epoch', fontsize=MINOR)
    plt.ylabel(y_label, fontsize=MINOR)
    plt.text(
        0.8, 0.75,
        f'Final: {val[-1]:.3e}',
        fontsize=MINOR,
        transform=plt.gca().transAxes
    )

    if log_y:
        plt.yscale('log')

    plt.legend(fontsize=MAJOR)
    plt.savefig(f'{plots_dir}{name}.png')


def plot_saliency(plots_dir: str, data: ndarray[floating], saliency: ndarray[floating]) -> None:
    """
    Plots the saliency and input for multiple saliency maps

    Parameters
    ----------
    plots_dir : str
        Directory to save plots
    data : (H,W) ndarray[floating]
        Input image of height H and width W
    saliency : (C,H,W) ndarray[floating]
        C saliency maps with height H and width W
    """
    i: int
    maximum: float
    datum: ndarray[floating]
    axes: dict[int | str, Axes] = _init_subplots(subplot_grid(saliency.shape[0] + 1))[0]
    axis: Axes

    for i, (datum, axis) in enumerate(zip(
            np.concatenate((data[np.newaxis], saliency)),
            axes.values(),
    )):
        maximum = np.max(np.abs(datum))
        axis.imshow(
            datum,
            cmap='hot' if i == 0 else 'twilight',
            vmin=0 if i == 0 else -maximum,
            vmax=maximum,
        )
        axis.set_title('Input' if i == 0 else f'Dim {i}', fontsize=MAJOR)
        axis.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)

    plt.savefig(f'{plots_dir}Saliency.png')
