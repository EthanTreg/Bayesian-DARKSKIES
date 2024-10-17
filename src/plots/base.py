"""
Base plotting class for other plots to build upon
"""
import logging
from typing import Any

import numpy as np
import scienceplots  # pylint: disable=unused-import
import matplotlib.pyplot as plt
from numpy import ndarray
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.legend import Legend
from matplotlib.contour import ContourSet
from matplotlib.colors import to_rgba_array
from matplotlib.figure import Figure, FigureBase
from matplotlib.collections import PathCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

from src.plots import utils

plt.style.use(["science", "grid", 'no-latex'])


class BasePlot:
    """
    Base class for creating plots

    Attributes
    ----------
    plots : list[Artist], default = []
        Plot artists
    axes : dict[int | str, Axes] | (R,C) ndarray | Axes
        Plot axes for R rows and C columns
    subfigs : (H,W) ndarray | None, default = None
        Plot sub-figures for H rows and W columns
    fig : Figure
        Plot figure
    legend : Legend | None, default = None
        Plot legend

    Methods
    -------
    savefig(path, name='')
        Saves the plot to the specified path
    subfigs(subfigs, titles=None, x_labels=None, y_labels=None, **kwargs)
        Initialises sub figures
    subplots(subplots, titles=None, fig=None, **kwargs)
        Generates subplots within a figure or sub-figure
    create_legend(rows=1, loc='outside upper center')
        Plots the legend
    plot_density(colour, data, ranges, axis, hatch='', order=None, confidences=None, **kwargs)
        Plots a density contour plot
    plot_histogram(colour, data, axis, log=False, norm=False, orientation='vertical', hatch='',
            range_=None, **kwargs)
        Plots a histogram or density plot
    plot_param_pairs(colour, data, hatch='', ranges=None, markers=None, **kwargs)
        Plots a pair plot to compare the distributions and comparisons between parameters
    plot_reconstruction(colour, x_data, pred, target, axis, uncertainty=None, major_axis=None)
        Plots reconstructions and residuals
    """
    def __init__(
            self,
            data: Any,
            density: bool = False,
            bins: int = 100,
            title: str = '',
            x_label: str = '',
            y_label: str = '',
            labels: list[str] | None = None,
            colours: list[str] | None = None,
            fig_size: tuple[int, int] = utils.RECTANGLE,
            **kwargs: Any):
        """
        Parameters
        ----------
        data : Any
            Data for the plot
        density : bool, default = False
            If the plot should plot contours and interpolate histograms
        bins : int, default = 200
            Number of bins for histograms or density interpolation
        title : str, default = ''
            Title of the plot
        x_label : str, default = ''
            X-label of the plot
        y_label : str, default = ''
            Y-label of the plot
        labels : list[str] | None, default = None
            Labels for the data to plot the legend
        colours : list[str] | None, default = XKCD_COLORS
            Colours for the data
        fig_size : tuple[int, int], default = RECTANGLE
            Size of the figure

        **kwargs
            Optional keyword arguments to pass to create_legend
        """
        self._density: bool = density
        self._bins: int = bins
        self._alpha: float = 0.6
        self._alpha_2d: float = 0.4
        self._default_name: str = 'base'
        self._labels: list[str] | None = labels
        self._colours: list[str] = colours or utils.COLOURS
        self._data: Any = data
        self.plots: list[Artist] = []
        self.axes: dict[int | str, Axes] | ndarray | Axes
        self.subfigs: ndarray | None = None
        self.fig: Figure = plt.figure(constrained_layout=True, figsize=fig_size)
        self.legend: Legend | None = None

        self.fig.suptitle(title, fontsize=utils.MAJOR)
        self.fig.supxlabel(x_label, fontsize=utils.MAJOR)
        self.fig.supylabel(y_label, fontsize=utils.MAJOR)
        self._axes_init()
        self._post_init()
        self._plot_data()

        if self._labels is not None and self._labels[0]:
            self.create_legend(**kwargs)

    @staticmethod
    def _multi_param_func_calls(
            func: str,
            objs: list[object] | ndarray,
            *args: list[Any] | None,
            kwargs: list[dict[str, Any]] | dict[str, Any] | None = None) -> None:
        """
        Calls a function for multiple objects with different arguments and keyword arguments

        Parameters
        ----------
        func : str
            Name of the object's function to call
        objs : list[object] | (N) ndarray
            List of N objects to apply the function to

        *args : list[Any] | None
            Arguments to pass to the function
        **kwargs : list[dict[str, Any]] | dict[str, Any] | None, default = None
            Optional keyword arguments to pass to the function
        """
        obj: object
        kwarg: dict[str, Any]
        arg: Any

        if len(args) != 0 and args[0] is None:
            return
        if len(args) == 0 or args[0] is None:
            args = [()] * len(objs)
        else:
            args = [tuple(arg) for arg in zip(*args)]

        if kwargs is None:
            kwargs = [{}] * len(objs)
        elif isinstance(kwargs, dict):
            kwargs = [kwargs] * len(objs)

        for obj, arg, kwarg in zip(objs, args, kwargs):
            getattr(obj, func)(*arg, **kwarg)

    def _axes_init(self) -> None:
        """
        Initialises the axes
        """
        self.axes = self.fig.gca()
        self.axes.tick_params(labelsize=utils.MINOR)

    def _post_init(self) -> None:
        """
        Performs any necessary post-initialisation tasks
        """

    def _plot_data(self) -> None:
        """
        Plots the data
        """

    def savefig(self, path: str, name: str = '') -> None:
        """
        Saves the plot to the specified path

        Parameters
        ----------
        path : str
            Path to save the plot
        name : str, default = ''
            Name of the plot, if empty, default name for the plot will be used
        """
        name = name or self._default_name
        name += '.png' if '.png' not in name else ''
        self.fig.savefig(f'{path}{name}')

    def subfigures(
            self,
            subfigs: tuple[int, int],
            titles: list[str] | None = None,
            x_labels: list[str] | None = None,
            y_labels: list[str] | None = None,
            **kwargs: Any) -> None:
        """
        Initialises sub figures

        Parameters
        ----------
        subfigs : tuple[int, int]
            Number of rows and columns for the sub figures
        titles : list[str] | None, default = None
            Title for each sub figure
        x_labels : list[str] | None, default = None
            X-label for each sub figure
        y_labels : list[str] | None, default = None
            Y-label for each sub figure

        **kwargs
            Optional arguments for the subfigures function
        """
        self.subfigs = self.fig.subfigures(*subfigs, **kwargs)
        self._multi_param_func_calls(
            'suptitle',
            self.subfigs.flatten(),
            titles,
            kwargs={'fontsize': utils.MAJOR},
        )
        self._multi_param_func_calls(
            'supxlabel',
            self.subfigs.flatten(),
            x_labels,
            kwargs={'fontsize': utils.MAJOR},
        )
        self._multi_param_func_calls(
            'supylabel',
            self.subfigs.flatten(),
            y_labels,
            kwargs={'fontsize': utils.MAJOR},
        )

    def subplots(
            self,
            subplots: str | tuple[int, int] | list[list[int | str]] | ndarray,
            titles: list[str] | None = None,
            fig: FigureBase | None = None,
            **kwargs: Any) -> None:
        """
        Generates subplots within a figure or sub-figure

        Parameters
        ----------
        subplots : str | tuple[int, int] | list[list[int | str]] | (R,C) ndarray
            Parameters for subplots or subplot_mosaic for R rows and C columns
        titles : list[str] | None, default = None
            Titles for each axis
        fig : FigureBase | None, default = self.fig
            Figure or sub-figure to add subplots to

        **kwargs
            Optional kwargs to pass to subplots or subplot_mosaic
        """
        fig = fig or self.fig

        if isinstance(subplots, tuple):
            self.axes = fig.subplots(*subplots, **kwargs)
        else:
            self.axes = fig.subplot_mosaic(subplots, **kwargs)

        self._multi_param_func_calls(
            'set_title',
            self.axes.flatten() if isinstance(self.axes, ndarray) else self.axes.values(),
            titles,
            kwargs={'fontsize': utils.MAJOR},
        )
        self._multi_param_func_calls(
            'tick_params',
            self.axes.flatten() if isinstance(self.axes, ndarray) else self.axes.values(),
            kwargs={'labelsize': utils.MINOR},
        )

    def create_legend(
            self,
            rows: int = 1,
            loc: str | tuple[float, float] = 'outside upper center') -> None:
        """
        Plots the legend

        Parameters
        ----------
        rows : int, default = 1
            Number of rows for the legend
        loc : str | tuple[float, float], default = 'outside upper center'
            Location to place the legend
        """
        cols: int = np.ceil(len(self._labels) / rows)
        fig_size: float = self.fig.get_size_inches()[0] * self.fig.dpi
        legend_offset: float
        handles: list[tuple[Artist, ...]]
        handle: ndarray | Artist

        # Generate legend handles
        self.plots = [handle.legend_elements()[0][0] if isinstance(handle, ContourSet) else handle
                      for handle in self.plots]
        handles = [tuple(handle) for handle in np.array_split(self.plots, len(self._labels))]

        # Create legend
        self.legend = self.fig.legend(
            handles,
            self._labels,
            fancybox=False,
            ncol=cols,
            fontsize=utils.MAJOR,
            borderaxespad=0.2,
            loc=loc,
            handler_map={tuple: utils.UniqueHandlerTuple(ndivide=None)}
        )
        legend_offset = float(np.array(self.legend.get_window_extent())[0, 0])

        # Recreate legend if it overflows the figure with more rows
        if legend_offset < 0:
            self.legend.remove()
            rows = np.abs(legend_offset) * 2 // fig_size + 2
            self.create_legend(rows=rows, loc=loc)

        # Update handles to remove transparency if there isn't any hatching and set point size
        for handle in self.legend.legend_handles:
            if not hasattr(handle, 'get_hatch') or handle.get_hatch() is None:
                handle.set_alpha(1)

            if isinstance(handle, PathCollection):
                handle.set_sizes([100])

    def plot_density(
            self,
            colour: str,
            data: ndarray,
            ranges: ndarray,
            axis: Axes,
            hatch: str = '',
            order: list[int] | None = None,
            confidences: list[float] | None = None,
            **kwargs: Any) -> None:
        """
        Plots a density contour plot

        Parameters
        ---------
        colour : str
            Colour of the contour
        data : (N,2) ndarray
            N (x,y) data points to generate density contour for
        ranges : (2,2) ndarray
            Min and max values for the x and y axes
        axis : Axes
            Axis to add density contour
        hatch : str, default = ''
            Hatching pattern for the contour
        order : list[int] | None, default = None
            Order of the axes, only required for 3D plots
        confidences : list[float] | None, default = [0.68]
            List of confidence values to plot contours for, starting with the lowest confidence

        **kwargs
            Optional kwargs to pass to Axes.contour and Axes.contourf
        """
        total: float
        levels: list[float]
        logger: logging.Logger = logging.getLogger(__name__)
        contour: ndarray
        grid: ndarray = np.mgrid[
                        ranges[0, 0]:ranges[0, 1]:self._bins * 1j,
                        ranges[1, 0]:ranges[1, 1]:self._bins * 1j,
                        ]
        kernel: gaussian_kde

        try:
            kernel = gaussian_kde(data.swapaxes(0, 1))
        except np.linalg.LinAlgError:
            logger.warning('Cannot calculate contours, skipping')
            return

        contour = np.reshape(kernel(grid.reshape(2, -1)).T, (self._bins, self._bins))
        total = np.sum(contour)
        levels = [np.max(contour)]

        if confidences is None:
            confidences = [0.68]

        for confidence in confidences:
            levels.insert(0, utils.contour_sig(total * confidence, contour))

        if levels[-1] == 0:
            logger.warning('Cannot calculate contours, skipping')
            return

        contour = np.concatenate((grid, contour[np.newaxis]), axis=0)

        if order is not None:
            contour = contour[order]

        self.plots.append(axis.contourf(
            *contour,
            levels,
            alpha=self._alpha_2d,
            colors=colour,
            hatches=[hatch],
            **kwargs,
        ))
        self.plots[-1]._hatch_color = tuple(to_rgba_array(colour, self._alpha)[0])
        self.plots.append(axis.contour(*contour, levels, colors=colour, **kwargs))

    def plot_histogram(
            self,
            colour: str,
            data: ndarray,
            axis: Axes,
            log: bool = False,
            norm: bool = False,
            orientation: str = 'vertical',
            hatch: str = '',
            range_: tuple[float, float] | None = None,
            **kwargs: Any) -> None:
        """
        Plots a histogram or density plot

        Parameters
        ----------
        colour : str
            Colour of the histogram or density plot
        data : (N) ndarray
            Data to plot for N data points
        axis : Axes
            Axis to plot on
        log : bool, default = False
            If data should be plotted on a log scale, expects linear data
        norm : bool, default = False
            If the histogram or density plot should be normalised to a maximum height of 1
        orientation : {'vertical', 'horizontal'}, default = 'vertical'
            Orientation of the histogram or density plot
        hatch : str, default = ''
            Hatching of the histogram or density plot
        range_ : tuple[float, float], default = None
            x-axis data range

        **kwargs
            Optional keyword arguments that get passed to Axes.plot and Axes.fill_between if
            self._density is True, else to Axes.hist
        """
        x_data: ndarray
        y_data: ndarray
        kernel: gaussian_kde
        axis.ticklabel_format(axis='y', scilimits=(-2, 2))

        if log:
            axis.set_xscale('log')

        if len(np.unique(data)) == 1:
            norm = True

        if range_ is None:
            range_ = (np.min(data), np.max(data))

        if self._density and len(np.unique(data)) > 1:
            kernel = gaussian_kde(np.log10(data) if log else data)
            x_data = np.linspace(*(np.log10(range_) if log else range_), self._bins)
            y_data = kernel(x_data)
            x_data = 10 ** x_data if log else x_data

            if norm:
                y_data /= np.max(y_data)

            if orientation == 'vertical':
                self.plots.append(axis.plot(
                    x_data,
                    y_data,
                    color=colour,
                    **kwargs,
                )[0])
                self.plots.append(axis.fill_between(
                    x_data,
                    y_data,
                    hatch=hatch,
                    fc=(colour, self._alpha_2d),
                    edgecolors=(colour, self._alpha),
                    **kwargs,
                ))
            else:
                self.plots.append(axis.plot(
                    y_data,
                    x_data,
                    color=colour,
                    **kwargs,
                )[0])
                self.plots.append(axis.fill_betweenx(
                    x_data,
                    y_data,
                    hatch=hatch,
                    fc=(colour, self._alpha_2d),
                    ec=(colour, self._alpha),
                    **kwargs,
                ))
        elif norm:
            y_data, x_data = np.histogram(
                data,
                np.logspace(*np.log10(range_), self._bins) if log else self._bins,
            )
            y_data = y_data / np.max(y_data)
            axis.bar(
                x_data[:-1],
                y_data,
                align='edge',
                width=np.diff(x_data),
                alpha=self._alpha_2d,
                color=colour,
                **kwargs
            )
        else:
            self.plots.append(axis.hist(
                data,
                lw=1,
                bins=np.logspace(*np.log10(range_), self._bins) if log else self._bins,
                hatch=hatch,
                histtype='stepfilled',
                orientation=orientation,
                fc=(colour, self._alpha_2d),
                ec=(colour, self._alpha),
                range=range_,
                **kwargs,
            )[-1][0])

    def plot_param_pairs(
            self,
            colour: str,
            data: ndarray,
            hatch: str = '',
            ranges: ndarray | None = None,
            markers: ndarray | None = None,
            **kwargs: Any) -> None:
        """
        Plots a pair plot to compare the distributions and comparisons between parameters

        Parameters
        ----------
        colour : str
            Colour of the data
        data : (N,L) ndarray
            Data to plot parameter pairs for N data points and L parameters
        hatch : str = ''
            Hatching of the histograms or density plots and contours
        ranges : (L,2) ndarray, default = None
            Ranges for L parameters, required if using kwargs to plot densities
        markers : (N) ndarray | None = None
            Markers for scatter plots for N data points

        **kwargs
            Optional keyword arguments passed to _plot_histogram and _plot_density
        """
        i: int
        j: int
        x_data: ndarray
        y_data: ndarray
        x_range: ndarray
        y_range: ndarray
        axes_row: ndarray
        axis: Axes

        if markers is None:
            markers = np.array(['.']) * len(data)

        data = np.swapaxes(data, 0, 1)

        # Loop through each subplot
        for i, (axes_row, y_data, y_range) in enumerate(zip(self.axes, data, ranges)):
            for j, (axis, x_data, x_range) in enumerate(zip(axes_row, data, ranges)):
                # Share y-axis for all scatter plots
                if i != j:
                    axis.sharey(axes_row[0])

                # Set number of ticks
                axis.locator_params(axis='x', nbins=3)
                axis.locator_params(axis='y', nbins=3)

                # Hide ticks for plots that aren't in the first column or bottom row
                if j != 0 or j == i:
                    axis.tick_params(labelleft=False, left=False)

                if i != self.axes.shape[0] - 1:
                    axis.tick_params(labelbottom=False, bottom=False)

                if j < i:
                    axis.set_xlim(x_range)
                    axis.set_ylim(y_range)

                # Plot scatter plots & histograms
                if i == j:
                    self.plot_histogram(
                        colour,
                        x_data,
                        axis,
                        hatch=hatch,
                        range_=x_range,
                        **kwargs,
                    )
                elif j < i:
                    for marker in np.unique(markers):
                        self.plots.append(axis.scatter(
                            x_data[marker == markers][:utils.SCATTER_NUM],
                            y_data[marker == markers][:utils.SCATTER_NUM],
                            s=4,
                            alpha=self._alpha,
                            color=colour,
                            marker=marker,
                        ))

                    if self._density:
                        self.plot_density(
                            colour,
                            np.stack((x_data, y_data), axis=1),
                            np.array((x_range, y_range)),
                            axis=axis,
                            hatch=hatch,
                            **kwargs,
                        )
                else:
                    axis.set_visible(False)

    def plot_reconstruction(
            self,
            colour: str,
            x_data: ndarray,
            pred: ndarray,
            target: ndarray,
            axis: Axes,
            uncertainty: ndarray | None = None,
            major_axis: Axes | None = None) -> Axes:
        """
        Plots reconstructions and residuals

        Parameters
        ----------
        colour : str
            Colour of the data
        x_data : (N) ndarray
            X-axis values for N data points
        pred : (N) ndarray
            Predicted values for N data points
        target : (N) ndarray
            Target values for N data points
        axis : Axes
            Axis to plot on
        uncertainty : (N) ndarray | None, default = None
            Uncertainties in the y predictions for N data points
        major_axis : Axes | None, default = None
            Major axis of the plot with the comparison

        Returns
        -------
        Axes
            Major axis of the plot with the comparison
        """
        major_axis = major_axis or make_axes_locatable(axis).append_axes('top', size='150%', pad=0)
        self.plots.append(axis.scatter(x_data, pred - target, marker='x', color=colour))
        self.plots.append(major_axis.scatter(x_data, pred, color=colour, marker='x'))

        # If the x-data is not the target data, then plot the target data
        if not (x_data == target).all():
            self.plots.append(major_axis.scatter(x_data, target))

        if uncertainty is not None:
            self.plots.append(major_axis.fill_between(
                x_data,
                pred - uncertainty,
                pred + uncertainty,
                alpha=self._alpha_2d,
                color=colour,
            ))
            self.plots.append(axis.fill_between(
                x_data,
                pred - uncertainty - target,
                pred + uncertainty - target,
                alpha=self._alpha_2d,
                color=colour,
            ))

        axis.set_ylabel('Residual', fontisze=utils.MAJOR)
        axis.hlines(0, xmin=np.min(x_data), xmax=np.max(x_data), color='k')
        major_axis.tick_params(axis='y', labelsize=utils.MINOR)
        major_axis.set_xticks([])
        return major_axis
