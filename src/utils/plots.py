"""
Creates several plots
"""
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import ndarray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from netloader.utils.utils import label_change

from src.utils.utils import legend_marker, subplot_grid

MAJOR = 24
MINOR = 20
SCATTER_NUM = 1000
COLOURS = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
CMAPS = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'YlOrBr', 'RdPu', 'Greys', 'YlGn', 'GnBu']
MARKERS = ['o', 'x', '^', 's', '*', '1', 'd', '+', 'p', 'D']
RECTANGLE = (16, 9)
SQUARE = (10, 10)
HI_RES = (32, 18)
HI_RES_SQUARE = (20, 20)


def _init_plot(
        subplots: str | tuple[int, int] | list | ndarray,
        x_label: str = None,
        y_label: str = None,
        fig_size: tuple[int, int] = RECTANGLE,
        **kwargs) -> tuple[Figure, dict | ndarray]:
    """
    Initialises subplots using either mosaic or subplots

    Parameters
    ----------
    subplots : str | tuple[int, int] | list | ndarray
        Argument for subplot or mosaic layout, mosaic will use str or list
        and subplots will use tuple
    legend : bool, default = False,
        If the figure will have a legend at the top, then space will be made
    x_label : str, default = None
        X label for the plot
    y_label : str, default = None
        Y label for the plot
    fig_size : tuple[int, int]
        Size of the figure
    **kwargs
        Optional arguments for the subplot or mosaic function

    Returns
    -------
    tuple[Figure, dict | ndarray]
        Figure and subplot axes
    """
    # Plots either subplot or mosaic
    if isinstance(subplots, tuple):
        fig, axes = plt.subplots(
            *subplots,
            figsize=fig_size,
            constrained_layout=True,
            **kwargs,
        )
    else:
        fig, axes = plt.subplot_mosaic(
            subplots,
            figsize=fig_size,
            layout='constrained',
            **kwargs,
        )

    if x_label:
        fig.supxlabel(x_label, fontsize=MAJOR)

    if y_label:
        fig.supylabel(y_label, fontsize=MAJOR)

    return fig, axes


def _legend(
        labels: list | ndarray,
        fig: Figure,
        columns: int = 2,
        loc: str = 'outside upper center') -> mpl.legend.Legend:
    """
    Plots a legend across the top of the plot

    Parameters
    ----------
    labels : (2,L) list | ndarray
        Legend matplotlib handles and labels of size L to be unpacked into handles and labels
    fig : Figure
        Figure to add legend to
    columns : int, default = 2
        Number of columns for the legend
    loc : str, default = 'outside upper center'
        Location to place the legend

    Returns
    -------
    Legend
        Legend object
    """
    fig_size = fig.get_size_inches()[0] * fig.dpi
    legend = fig.legend(
        *labels,
        loc=loc,
        ncol=columns,
        fontsize=MAJOR,
        borderaxespad=0.2,
    )

    legend_offset = np.array(legend.get_window_extent())[0, 0]

    if legend_offset < 0:
        legend.remove()
        rows = np.abs(legend_offset) // fig_size + 2
        columns = np.ceil(len(labels[1]) / rows)
        legend = fig.legend(
            *labels,
            loc=loc,
            ncol=columns,
            fontsize=MAJOR,
            borderaxespad=0.2,
        )

    for handle in legend.legend_handles:
        handle.set_alpha(1)

        if isinstance(handle, mpl.collections.PathCollection):
            handle.set_sizes([100])

    return legend


def _contour_sig(counts: float, contour: ndarray) -> float:
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


def _plot_clusters_2d(
        label: str,
        colour: str,
        data: ndarray,
        axes: ndarray,
        ranges: ndarray,
        density: bool = True,
        bins: int = 200,
        cmap: str | mpl.colors.Colormap = None,
        markers: ndarray = None):
    """
    Plots scattered data with contours and histograms to show data distribution for 2D data

    Parameters
    ----------
    colour : str
        Colour of the data
    label : str
        Label for the data
    data : (N,2) ndarray
        N (x,y) data points to plot
    axes : (2,2) ndarray
        Axes to plot histograms and the scatter plot
    ranges : (2,2) ndarray
        Min and max values for the x and y axes
    density : bool, default = True
        If density contours should be plotted or confidence ellipses
    bins : int, default = 200
        Resolution of the density plot contours or number of histogram bins
    cmap : str | Colormap, default = None
        Colour of the density contours, required if density is True
    markers : ndarray, default = None
        Markers for the data
    """
    scat_alpha = 0.2
    hist_alpha = 0.4

    _plot_histogram(
        data[:, 0],
        axes[0, 0],
        density=density,
        bins=bins,
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
        bins=bins,
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
        _plot_density(colour, ranges, data, axes[1, 0], cmap, bins=bins)
    else:
        _plot_ellipse(colour, data, axes[1, 0], stds=[1, 2])


def _plot_clusters_3d(
        label: str,
        colour: str,
        cmap: mpl.colors.Colormap,
        data: ndarray,
        ranges: ndarray,
        axis: Axes,
        bins: int = 200,
        markers: ndarray = None):
    """
    Plots scattered data with contours and histograms to show data distribution for 2D data

    Parameters
    ----------
    label : str
        Label for the data
    colour : str
        Colour of the data
    cmap : str | Colormap
        Colour of the density contours
    data : Nx3 ndarray
        N (x,y,z) data points to plot
    axis : Axes
        Axis to plot 3D scatter plot with contours
    ranges : 2x2 ndarray
        Min and max values for the x and y axes
    bins : int, default = 200
        Resolution of the density plot contours
    """
    alpha = 0.4
    axes = ['x', 'y', 'z']
    orders = [[0, 1, 2], [0, 2, 1], [2, 1, 0]]

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
            cmap,
            bins=bins,
            alpha=alpha,
            order=order,
            zdir=axes[order[-1]],
            offset=ranges[order[-1], 0],
        )


def _plot_density(
        colour: str,
        ranges: ndarray,
        data: ndarray,
        axis: Axes,
        cmap: str | mpl.colors.Colormap,
        bins: int = 200,
        alpha: float = 0.2,
        order: list[int] = None,
        confidences: list[float] = None,
        **kwargs):
    """
    Plots a density contour plot

    Parameters
    ---------
    colour : str
        Colour of the contour lines
    ranges : (2,2) ndarray
        Min and max values for the x and y axes
    data : (N,2) ndarray
        N (x,y) data points to generate density contour for
    axis : Axes
        Axis to add density contour
    cmap : str | Colormap
        Colour map for the density contours
    bins : int, default = 200
        Resolution of the contours
    alpha : float, default = 0.2
        Alpha value for the contour
    confidences : list[float], default = [0.68]
        List of confidence values to plot contours for, starting with the lowest confidence

    **kwargs
        Optional kwargs to pass to Axes.contour and Axes.contourf
    """
    grid = np.mgrid[ranges[0, 0]:ranges[0, 1]:bins * 1j, ranges[1, 0]:ranges[1, 1]:bins * 1j]

    try:
        kernel = gaussian_kde(data.swapaxes(0, 1))
    except np.linalg.LinAlgError:
        logging.warning('Cannot calculate contours, skipping')
        return

    contour = np.reshape(kernel(grid.reshape(2, -1)).T, (bins, bins))
    total = np.sum(contour)
    levels = [np.max(contour)]

    if 'density' in kwargs:
        del kwargs['density']

    if confidences is None:
        confidences = [0.68]

    for confidence in confidences:
        levels.insert(0, _contour_sig(total * confidence, contour))

    if levels[-1] == 0:
        logging.warning('Cannot calculate contours, skipping')
        return

    contour = np.concatenate((grid, contour[np.newaxis]), axis=0)

    if order is not None:
        contour = contour[order]

    axis.contourf(*contour, levels, alpha=alpha, cmap=cmap, **kwargs)
    axis.contour(*contour, levels, colors=colour, **kwargs)


def _plot_ellipse(colour: str, data: ndarray, axis: Axes, stds: list[int] = None):
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
        data: ndarray,
        axis: Axes,
        log: bool = False,
        density: bool = False,
        bins: int = 100,
        alpha: float = None,
        colour: str = 'blue',
        labels: list[str] = None,
        data_range: tuple[float, float] = None,
        data_twin: ndarray = None,
        **kwargs) -> None | Axes:
    """
    Plots a histogram subplot with twin data if provided

    Parameters
    ----------
    data : ndarray
        Primary data to plot
    axis : Axes
        Axis to plot on
    log : bool, default = False
        If data should be plotted on a log scale, expects linear data
    density : bool, default = False
        If histogram should be plotted as a kernel density estimation
    bins : int, default = 100
        Number of bins
    alpha : float, default = 0.2 if density, 0.5 if data_twin is provided; otherwise, 1
        Transparency of the histogram, gets halved if data_twin is provided
    colour : str
        Colour of the histogram or density plot
    labels : list[str], default = None
        Labels for data and, if provided, data_twin
    data_range : tuple[float, float], default = None
        x-axis data range, required if density is True
    data_twin : ndarray, default = None
        Secondary data to plot

    **kwargs
        Optional keyword arguments that get passed to Axes.plot and Axes.fill_between if density is
        True, else to Axes.hist

    Returns
    -------
    None | Axes
        If data_twin is provided, returns twin axis
    """
    bin_num = bins

    if not labels:
        labels = ['', '']

    if alpha is None and density:
        alpha = 0.2
    elif alpha is None and data_twin is None:
        alpha = 1
    else:
        alpha = 0.5

    if 'cmap' in kwargs:
        del kwargs['cmap']

    if log:
        bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), bin_num)
        axis.set_xscale('log')

    if density:
        kernel = gaussian_kde(data)
        x_data = np.linspace(*data_range, bins)
        y_data = kernel(x_data)

        if 'orientation' in kwargs and kwargs['orientation'] == 'horizontal':
            del kwargs['orientation']
            axis.plot(y_data, x_data, label=labels[0], color=colour, **kwargs)
            axis.fill_betweenx(x_data, y_data, alpha=alpha, color=colour, **kwargs)
        else:
            axis.plot(x_data, y_data, label=labels[0], color=colour, **kwargs)
            axis.fill_between(x_data, y_data, alpha=alpha, color=colour, **kwargs)
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
            bins=bin_num,
            alpha=alpha,
            colour='orange',
            labels=[labels[1]],
            **kwargs,
        )
        return twin_axis

    return None


def plot_clusters(
        path: str,
        classes: ndarray,
        data: ndarray,
        density: bool = True,
        plot_3d: bool = False,
        bins: int = 200,
        labels: list[str] = None,
        predictions: ndarray = None):
    """
    Plots clusters either as a 2D scatter plot or 1D histogram

    Parameters
    ----------
    path : str
        Path to save plots
    classes : N ndarray
        Data classes for N data points
    data : NxD ndarray
        N cluster data points of dimension D = {1,2}
    density : bool, default = True
        If density contours should be plotted or confidence ellipses
    plot_3d : bool, default = False
        If 3D plot or corner plot should be used for 3D data
    bins : int, default = 200
        Resolution of the density plot contours or number of bins if density is False
    labels : list[str], default = None
        Class labels
    predictions : N ndarray, default = None
        Class predicted labels
    """
    columns = len(labels)
    pad = 0.05
    markers = None
    ranges = np.stack((
        np.min(data - np.abs(pad * data), axis=0),
        np.max(data + np.abs(pad * data), axis=0),
    ), axis=1)

    if data.shape[1] == 1:
        fig = plt.figure(figsize=RECTANGLE, constrained_layout=True)
    elif data.shape[1] == 2:
        fig, axes = _init_plot(
            (2, 2),
            sharex='col',
            sharey='row',
            width_ratios=[3, 1],
            height_ratios=[1, 3],
        )
        axes[0, 1].remove()
        axes[0, 0].tick_params(bottom=False)
        axes[1, 1].tick_params(left=False)
        axis = axes[1, 0]
    elif data.shape[1] == 3 and plot_3d:
        columns = 3
        fig = plt.figure(figsize=SQUARE, constrained_layout=True)
        axis = fig.add_subplot(projection='3d')
        axis.set_xlim(ranges[0])
        axis.set_ylim(ranges[1])
        axis.set_zlim(ranges[2])
        axis.invert_yaxis()
    else:
        fig, axes = _init_plot((data.shape[1],) * 2, fig_size=HI_RES, sharex='col')

    if labels is None:
        labels = [None] * np.unique(classes).size

    if predictions is not None:
        legend_labels = legend_marker(COLOURS, labels, MARKERS)
    else:
        legend_labels = legend_marker(COLOURS, labels)

    # Plot each cluster class
    for class_, colour, cmap, label in zip(np.unique(classes), COLOURS, CMAPS, labels):
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
                labels=[label],
                data_range=(np.min(data), np.max(data)),
            )
        elif data.shape[1] == 2:
            _plot_clusters_2d(
                label,
                colour,
                class_data,
                axes,
                ranges,
                density=density,
                bins=bins,
                cmap=cmap,
                markers=markers,
            )
        elif data.shape[1] == 3 and plot_3d:
            _plot_clusters_3d(
                label,
                colour,
                cmap,
                class_data,
                ranges,
                axis,
                bins=bins,
                markers=markers,
            )
        else:
            plot_param_pairs(
                class_data,
                ranges=ranges,
                axes=axes,
                density=density,
                bins=bins,
                colour=colour,
                cmap=cmap,
            )

    if data.shape[1] > 1 and labels[0] is not None:
        _legend(legend_labels, fig, columns=columns)

    plt.savefig(f'{path}.png')


def plot_confusion(plots_dir: str, labels: list[str], targets: ndarray, predictions: ndarray):
    """
    Plots the confusion matrix between targets and predictions

    Parameters
    ----------
    plots_dir : str
        Directory to save plots
    labels : list[str]
        Labels for each class
    targets : N ndarray
        Target values
    predictions : N ndarray
        Predicted values
    """
    classes = np.unique(targets)
    matrix = np.zeros((len(classes), len(classes)))
    plt.figure(figsize=SQUARE if len(classes) < 5 else HI_RES_SQUARE, constrained_layout=True)

    # Generate confusion matrix
    for matrix_row, class_ in zip(matrix, classes):
        idxs = targets == class_
        class_predicts, counts = np.unique(predictions[idxs], return_counts=True)
        class_predicts = label_change(class_predicts, classes)
        matrix_row[class_predicts] = counts / np.count_nonzero(idxs) * 100

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
        data: list[ndarray] | ndarray,
        y_axis: bool = True,
        num_plots: int = 12,
        labels: list[str] = None,
        titles: list[str] = None,
        data_twin: ndarray = None,
        **kwargs):
    """
    Plots the distributions for a number of examples

    Parameters
    ----------
    plots_dir : str
        Directory to save plots
    name : str
        File name to save plot
    data : list[ndarray] | ndarray
        Distributions to plot, each row is a different distribution
    y_axis : bool, default = True
        If y-axis should be plotted
    num_plots : int, default = 16
        Number of distributions to plot, number of rows in data will be used if rows < num_plots
    labels : list[str], default = None
        Labels for data and data_twin if provided
    titles : list[str], default = None
        Titles for the distributions
    data_twin : ndarray, default = None
        Twin distributions to plot, each row is a different distribution corresponding to data

    **kwargs
        Optional keyword arguments for plotting the histogram
    """
    fig, axes = _init_plot(subplot_grid(min(len(data), num_plots)))

    if labels is None:
        labels = (None, None)

    if data_twin is None:
        data_twin = [None] * len(data)

    if titles is None:
        titles = [None] * len(data)

    for title, datum, datum_twin, axis in zip(titles, data, data_twin, axes.values()):
        twin_axis = _plot_histogram(
            datum,
            axis,
            labels=labels,
            data_twin=datum_twin,
            **kwargs,
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

    if labels[0] is not None:
        _legend(labels, fig)

    plt.savefig(f'{plots_dir}{name}.png')


def plot_param_comparison(plots_dir: str, x_data: ndarray, y_data: ndarray):
    """
    Plots y_data against x_data for comparison

    Parameters:
    ----------
    plots_dir : str
        Directory to save plots
    x_data : ndarray
        X-axis data
    y_data : ndarray
        Y-axis data
    """
    value_range = [np.min(x_data), np.max(x_data)]
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

    plt.savefig(f'{plots_dir}Parameter_Comparison.png', transparent=False)


def plot_param_pairs(
        data: ndarray,
        plots_dir: str = None,
        ranges: ndarray = None,
        axes: ndarray = None,
        **kwargs):
    """
    Plots a pair plot to compare the distributions and comparisons between parameters

    Parameters
    ----------
    data : NxL ndarray
        Data to plot parameter pairs for N data points and L parameters
    plots_dir : str, default = None
        Directory to save plots
    ranges : Lx2 ndarray, default = None
        Ranges for L parameters, required if using kwargs to plot densities
    axes : LxL ndarray, default = None
        Axes to use for plotting L parameters

    **kwargs
        Optional keyword arguments passed to _plot_histogram and _plot_density
    """
    # Initialize pair plots
    data = np.swapaxes(data, 0, 1)

    if ranges is None:
        ranges = [None] * data.shape[0]

    if axes is None:
        _, axes = _init_plot((data.shape[0],) * 2, sharex='col')

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
                _plot_histogram(x_data, axis, data_range=x_range, **kwargs)
                axis.tick_params(labelleft=False, left=False)
            elif j < i:
                axis.scatter(
                    x_data[:SCATTER_NUM],
                    y_data[:SCATTER_NUM],
                    s=4,
                    alpha=0.2,
                )

                if 'density' in kwargs and kwargs['density']:
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
        val: list,
        log_y: bool = True,
        train: list = None):
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
    val : list
        Validation performance
    log_y : bool, default = True
        If y-axis should be logged
    train : list, default = None
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

    legend = plt.legend(fontsize=MAJOR)
    legend.get_frame().set_alpha(None)
    plt.savefig(f'{plots_dir}{name}.png', transparent=False)


def plot_saliency(plots_dir: str, data: ndarray, saliency: ndarray) -> None:
    """
    Plots the saliency and input for multiple saliency maps

    Parameters
    ----------
    plots_dir : str
        Directory to save plots
    data : (H,W) ndarray
        Input image of height H and width W
    saliency : (C,H,W)
        C saliency maps with height H and width W
    """
    _, axes = _init_plot(subplot_grid(saliency.shape[0] + 1))

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

