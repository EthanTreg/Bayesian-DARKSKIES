"""
Creates several plots
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import ndarray
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.utils.utils import subplot_grid

MAJOR = 24
MINOR = 20
FIG_SIZE = (16, 9)
SCATTER_NUM = 1000


def _init_plot(
        subplots: str | tuple[int, int] | list | ndarray,
        legend: bool = False,
        x_label: str = None,
        y_label: str = None,
        plot_kwargs: dict = None) -> tuple[Figure, dict | ndarray]:
    """
    Initialises subplots using either mosaic or subplots

    Parameters
    ----------
    subplots : string | tuple[integer, integer] | list | ndarray
        Argument for subplot or mosaic layout, mosaic will use string or list
        and subplots will use tuple
    legend : boolean, default = False,
        If the figure will have a legend at the top, then space will be made
    x_label : string, default = None
        X label for the plot
    y_label : string, default = None
        Y label for the plot
    plot_kwargs : dict, default = None
        Optional arguments for the subplot or mosaic function, excluding gridspec_kw

    Returns
    -------
    tuple[Figure, dictionary | ndarray]
        Figure and subplot axes
    """
    text_offset = 0.04
    bbox = [0, 0, 1, 1]

    if not plot_kwargs:
        plot_kwargs = {}

    # BBox for optional layouts
    if legend:
        bbox[3] -= 2 * text_offset

    if x_label:
        bbox[1] += text_offset

    if y_label:
        bbox[0] += text_offset

    # Plots either subplot or mosaic
    if isinstance(subplots, tuple):
        fig, axes = plt.subplots(
            *subplots,
            figsize=FIG_SIZE,
            constrained_layout=True,
            **plot_kwargs,
        )
    else:
        fig, axes = plt.subplot_mosaic(
            subplots,
            figsize=FIG_SIZE,
            constrained_layout=True,
            **plot_kwargs,
        )

    fig.get_layout_engine().set(rect=bbox)

    if x_label:
        plt.figtext(0.5, 0.02, x_label, ha='center', va='center', fontsize=MAJOR)

    if y_label:
        plt.figtext(
            0.02,
            0.5,
            y_label,
            ha='center',
            va='center',
            rotation='vertical',
            fontsize=MAJOR,
        )

    return fig, axes


def _legend(labels: list | ndarray, columns: int = 2) -> matplotlib.legend.Legend:
    """
    Plots a legend across the top of the plot

    Parameters
    ----------
    labels : list | ndarray
        Legend matplotlib handles and labels as an array to be unpacked into handles and labels
    columns : integer, default = 2
        Number of columns for the legend

    Returns
    -------
    Legend
        Legend object
    """
    legend = plt.figlegend(
        *labels,
        loc='lower center',
        ncol=columns,
        bbox_to_anchor=(0.5, 0.91),
        fontsize=MAJOR,
    )
    legend.get_frame().set_alpha(None)

    for handle in legend.legendHandles:
        if isinstance(handle, matplotlib.collections.PathCollection):
            handle.set_sizes([100])

    return legend


def _plot_histogram(
        data: ndarray,
        axis: Axes,
        log: bool = False,
        labels: list[str] = None,
        hist_kwargs: dict = None,
        data_twin: ndarray = None) -> None | Axes:
    """
    Plots a histogram subplot with twin data if provided

    Parameters
    ----------
    data : ndarray
        Primary data to plot
    axis : Axes
        Axis to plot on
    log : boolean, default = False
        If data should be plotted on a log scale, expects linear data
    labels : list[string], default = None
        Labels for data and, if provided, data_twin
    hist_kwargs : dictionary, default = None
        Optional keyword arguments for plotting the histogram
    data_twin : ndarray, default = None
        Secondary data to plot

    Returns
    -------
    None | Axes
        If data_twin is provided, returns twin axis
    """
    bins = bin_num = 100

    if not labels:
        labels = ['', '']

    if not hist_kwargs:
        hist_kwargs = {}

    if log:
        bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), bin_num)
        axis.set_xscale('log')

    axis.hist(data, bins=bins, alpha=0.5, label=labels[0], **hist_kwargs)
    axis.tick_params(labelsize=MINOR)
    axis.ticklabel_format(axis='y', scilimits=(-2, 2))

    if data_twin is not None:
        twin_axis = axis.twinx()
        _plot_histogram(
            data_twin,
            twin_axis,
            log=log,
            labels=[labels[1]],
            hist_kwargs={'color': 'orange'} | hist_kwargs,
        )
        return twin_axis

    return None


def plot_performance(
        plots_dir: str,
        name: str,
        y_label: str,
        val: list,
        log_y: bool = True,
        train: list = None):
    """
    Plots training and validation performance as a function of epochs

    Parameters
    ----------
    plots_dir : string
        Directory to save plots
    name : string
        File name to save plot
    y_label : string
        Performance metric
    val : list
        Validation performance
    log_y : boolean, default = True
        If y-axis should be logged
    train : list, default = None
        Training performance
    """
    plt.figure(figsize=FIG_SIZE, constrained_layout=True)

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

    legend = plt.legend(fontsize=MAJOR)
    legend.get_frame().set_alpha(None)
    plt.savefig(f'{plots_dir}{name}.png', transparent=False)


def plot_param_comparison(plots_dir: str, x_data: ndarray, y_data: ndarray):
    """
    Plots y_data against x_data for comparison

    Parameters:
    ----------
    plots_dir : string
        Directory to save plots
    x_data : ndarray
        X-axis data
    y_data : ndarray
        Y-axis data
    """
    value_range = [np.min(x_data), np.max(x_data)]
    plt.figure(figsize=FIG_SIZE, constrained_layout=True)
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
    # axis.set_xscale('log')
    # axis.set_yscale('log')

    plt.savefig(f'{plots_dir}Parameter_Comparison.png', transparent=False)


def plot_distributions(
        plots_dir: str,
        name: str,
        data: ndarray,
        y_axis: bool = True,
        num_plots: int = 12,
        labels: list[str] = None,
        hist_kwargs: dict = None,
        titles: ndarray = None,
        data_twin: ndarray = None):
    """
    Plots the distributions for a number of examples

    Parameters
    ----------
    plots_dir : string
        Directory to save plots
    name : string
        File name to save plot
    data : ndarray
        Distributions to plot, each row is a different distribution
    y_axis : boolean, default = True
        If y-axis should be plotted
    num_plots : integer, default = 16
        Number of distributions to plot, number of rows in data will be used if rows < num_plots
    labels : list[string], default = None
        Labels for data and data_twin if provided
    hist_kwargs : dictionary, default = None
        Optional keyword arguments for plotting the histogram
    titles : ndarray, default = None
        Titles for the distributions
    data_twin : ndarray, default = None
        Twin distributions to plot, each row is a different distribution corresponding to data
    """
    _, axes = _init_plot(subplot_grid(min(data.shape[0], num_plots)), legend=True)

    if labels is None:
        labels = (None, None)

    if data_twin is None:
        data_twin = [None] * data.shape[0]

    if titles is None:
        titles = [None] * data.shape[0]

    for title, datum, datum_twin, axis in zip(titles, data, data_twin, axes.values()):
        twin_axis = _plot_histogram(
            datum,
            axis,
            labels=labels,
            hist_kwargs=hist_kwargs,
            data_twin=datum_twin,
        )
        axis.set_title(title, fontsize=MINOR)

        if not y_axis:
            axis.tick_params(labelleft=False, left=False)

        if twin_axis and not y_axis:
            twin_axis.tick_params(labelright=False, right=False)

    if twin_axis:
        labels = np.hstack((
            axes[0].get_legend_handles_labels(),
            twin_axis.get_legend_handles_labels(),
        ))
    elif labels:
        labels = axes[0].get_legend_handles_labels()

    if labels is not None:
        _legend(labels)

    plt.savefig(f'{plots_dir}{name}.png')
