"""
Cluster plotting classes
"""
from typing import Any

import numpy as np
from numpy import ndarray
from matplotlib.axes import Axes

from src.plots import utils
from src.plots.base import BasePlot


class _BasePlotClusters(BasePlot):
    """
    Base class for plotting clusters

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
    """
    def __init__(
            self,
            data: ndarray,
            targets: ndarray,
            density: bool = False,
            bins: int = 200,
            labels: list[str] | None = None,
            colours: list[str] | None = None,
            hatches: list[str] = None,
            fig_size: tuple[int, int] = utils.RECTANGLE,
            preds: ndarray | None = None):
        """
        Parameters
        ----------
        data : (N,D) ndarray
            Data of size N and D dimensions to plot
        targets : (N) ndarray
            Target classes for the data
        density : bool, default = False
            If data should be plotted with density contours and histograms smoothed out
        bins : int, default = 200
            Number of bins for the histogram or number of steps in the contour interpolation
        labels : list[str] | None, default = None
            Labels for each class in targets
        colours : list[str] | None, default = XKCD_COLORS
            Colours for each class in targets
        hatches : list[str] | None, default = None
            Hatches for each class in targets
        fig_size : tuple[int, int], default = RECTANGLE
            Size of the figure
        preds : ndarray | None, default = None
            Predictions to assign markers to each predicted class
        """
        class_: float
        pad: float = 0.05
        marker: str
        self._hatches: list[str] = hatches or [''] * len(np.unique(targets))
        self._data: ndarray
        self._targets: ndarray = targets
        self._markers: ndarray = np.array([utils.MARKERS[0]] * len(data))
        self._preds: ndarray | None = preds
        self._ranges: ndarray = np.stack((
            np.min(data - np.abs(pad * data), axis=0),
            np.max(data + np.abs(pad * data), axis=0),
        ), axis=1)

        if self._preds is not None:
            for class_, marker in zip(np.unique(self._targets), utils.MARKERS):
                self._markers[self._preds == class_] = marker

        super().__init__(
            data,
            density=density,
            bins=bins,
            labels=labels,
            colours=colours,
            fig_size=fig_size,
        )

    def _post_init(self):
        self._default_name = 'clusters'
        self._contour_alpha = max(0.2, 1 / len(np.unique(self._targets)))

    def _plot_data(self) -> None:
        """
        Plots the data for each class
        """
        class_: float
        hatch: str
        colour: str
        idxs: ndarray

        for class_, colour, hatch in zip(np.unique(self._targets), self._colours, self._hatches):
            idxs = self._targets == class_
            self._plot_clusters(colour, self._data[idxs], self._markers[idxs], hatch=hatch)

    def _plot_clusters(
            self,
            colour: str,
            data: ndarray,
            markers: ndarray,
            hatch: str = '') -> None:
        """
        Plots the data for a class

        Parameters
        ----------
        colour : str
            Colour for the class
        data : ndarray
            Data for the class
        markers : ndarray
            Markers for the predictions of the class
        hatch : str, default = ''
            Hatching pattern for the class
        """


class _PlotClusters1D(_BasePlotClusters):
    """
    Plots 1D cluster data as a histogram

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
    """
    def _plot_clusters(
            self,
            colour: str,
            data: ndarray,
            _: ndarray,
            hatch: str = '') -> None:
        """
        Plots the 1D data for a class

        Parameters
        ----------
        colour : str
            Colour for the class
        data : ndarray
            Data for the class
        hatch : str, default = ''
            Hatching pattern for the class
        """
        self.plot_histogram(
            colour,
            data.flatten(),
            self.axes,
            hatch=hatch,
            range_=self._ranges[0],
        )


class _PlotClusters2D(_BasePlotClusters):
    """
    Plots 2D cluster data with a wider scatter plot and horizontal histogram for dimension 2

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
    """
    def _axes_init(self) -> None:
        self.subplots(
            (2, 2),
            sharex='col',
            sharey='row',
            width_ratios=[3, 1],
            height_ratios=[1, 3],
        )
        self.axes[0, 1].remove()
        self.axes[0, 0].tick_params(bottom=False)
        self.axes[1, 1].tick_params(left=False)
        self.axes[0, 1].tick_params(labelsize=utils.MINOR)

    def _plot_clusters(
            self,
            colour: str,
            data: ndarray,
            markers: ndarray,
            hatch: str = '') -> None:
        assert isinstance(self.axes, ndarray)
        self.plot_histogram(
            colour,
            data[:, 0],
            self.axes[0, 0],
            hatch=hatch,
            range_=self._ranges[0],
        )
        self.axes[1, 0].set_xlim(self._ranges[0])
        self.axes[1, 0].set_ylim(self._ranges[1])
        self.plot_histogram(
            colour,
            data[:, 1],
            self.axes[1, 1],
            hatch=hatch,
            range_=self._ranges[1],
            orientation='horizontal',
        )

        for marker in np.unique(markers):
            self.plots.append(self.axes[1, 0].scatter(
                *data[marker == markers].swapaxes(0, 1),
                color=colour,
                marker=marker,
            ))

        if self._density:
            self.plot_density(colour, data, self._ranges, self.axes[1, 0], hatch=hatch)


class _PlotClusters3D(_BasePlotClusters):
    """
    Plots 3D cluster data as a 3D plot

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
    """
    def __init__(
            self,
            data: ndarray,
            targets: ndarray,
            density: bool = False,
            bins: int = 200,
            labels: list[str] | None = None,
            colours: list[str] | None = None,
            hatches: list[str] | None = None,
            fig_size: tuple[int, int] = utils.SQUARE,
            preds: ndarray | None = None):
        """
        Parameters
        ----------
        data : (N,D) ndarray
            Data of size N and D dimensions to plot
        targets : (N) ndarray
            Target classes for the data
        density : bool, default = False
            If data should be plotted with density contours and histograms smoothed out
        bins : int, default = 200
            Number of bins for the histogram or number of steps in the contour interpolation
        labels : list[str] | None, default = None
            Labels for each class in targets
        colours : list[str] | None, default = XKCD_COLORS
            Colours for each class in targets
        hatches : list[str] | None, default = None
            Hatches for each class in targets
        fig_size : tuple[int, int], default = SQUARE
            Size of the figure
        preds : ndarray | None, default = None
            Predictions to assign markers to each predicted class
        """
        super().__init__(
            data,
            targets,
            density=density,
            bins=bins,
            labels=labels,
            colours=colours,
            hatches=hatches,
            preds=preds,
            fig_size=fig_size,
        )

    def _axes_init(self) -> None:
        self._axes = self.fig.add_subplot(projection='3d')
        self._axes.set_xlim(self._ranges[0])
        self._axes.set_xlim(self._ranges[1])
        self._axes.set_xlim(self._ranges[2])
        self._axes.invert_yaxis()
        self._axes.tick_params(labelsize=utils.MINOR)

    def _plot_clusters(
            self,
            colour: str,
            data: ndarray,
            markers: ndarray,
            hatch: str = '') -> None:
        marker: str
        order: list[int]
        axes: list[str] = ['x', 'y', 'z']
        orders: list[list[int]] = [[0, 1, 2], [0, 2, 1], [2, 1, 0]]

        for marker in np.unique(markers):
            assert isinstance(self._axes, Axes)
            self.plots.append(self._axes.scatter(
                *data[marker == markers].swapaxes(0, 1),
                color=colour,
                marker=marker,
            ))

        if self._density:
            for order in orders:
                self.plot_density(
                    colour,
                    data[:, order[:2]],
                    self._ranges[order[:2]],
                    self._axes,
                    hatch=hatch,
                    order=order,
                    zdir=axes[order[-1]],
                    offset=self._ranges[order[-1], 0],
                )


class _PlotClustersND(_BasePlotClusters):
    """
    Plots N number of cluster dimensions as a pair plot

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
    """
    def __init__(
            self,
            data: ndarray,
            targets: ndarray,
            density: bool = False,
            bins: int = 200,
            labels: list[str] | None = None,
            colours: list[str] | None = None,
            hatches: list[str] | None = None,
            fig_size: tuple[int, int] = utils.HI_RES,
            preds: ndarray | None = None):
        """
        Parameters
        ----------
        data : (N,D) ndarray
            Data of size N and D dimensions to plot
        targets : (N) ndarray
            Target classes for the data
        density : bool, default = False
            If data should be plotted with density contours and histograms smoothed out
        bins : int, default = 200
            Number of bins for the histogram or number of steps in the contour interpolation
        labels : list[str] | None, default = None
            Labels for each class in targets
        colours : list[str] | None, default = XKCD_COLORS
            Colours for each class in targets
        hatches : list[str] | None, default = None
            Hatches for each class in targets
        fig_size : tuple[int, int], default = HI_RES
            Size of the figure
        preds : ndarray | None, default = None
            Predictions to assign markers to each predicted class
        """
        super().__init__(
            data,
            targets,
            density=density,
            bins=bins,
            labels=labels,
            colours=colours,
            hatches=hatches,
            preds=preds,
            fig_size=fig_size,
        )

    def _axes_init(self) -> None:
        self.subplots((self._data.shape[1],) * 2, sharex='col')

    def _plot_clusters(
            self,
            colour: str,
            data: ndarray,
            markers: ndarray,
            hatch: str = '') -> None:
        self.plot_param_pairs(colour, data, markers=markers, hatch=hatch, ranges=self._ranges)


class PlotClusters:
    """
    Plots data with contours as a histogram for 1D data, 3D plot for 3D data, or a pair plot for ND
    data
    """
    def __new__(
            cls,
            data: ndarray,
            targets: ndarray,
            density: bool = False,
            plot_3d: bool = False,
            bins: int = 200,
            labels: list[str] | None = None,
            colours: list[str] | None = None,
            hatches: list[str] | None = None,
            preds: ndarray | None = None,
            **kwargs: Any):
        """
        Parameters
        ----------
        data : (N,D) ndarray
            Data of size N and D dimensions to plot
        targets : (N) ndarray
            Target classes for the data
        density : bool, default = False
            If data should be plotted with density contours and histograms smoothed out
        plot_3d : bool, default = False
            If 3D data should be plotted as a 3D plot or as a pair plot
        bins : int, default = 200
            Number of bins for the histogram or number of steps in the contour interpolation
        labels : list[str] | None, default = None
            Labels for each class in targets
        colours : list[str] | None, default = XKCD_COLORS
            Colours for each class in targets
        hatches : list[str] | None, default = None
            Hatches for each class in targets
        preds : ndarray | None, default = None
            Predictions to assign markers to each predicted class

        **kwargs
            Optional keyword arguments to pass to cluster plotting
        """
        if data.shape[1] == 1:
            return _PlotClusters1D(
                data,
                targets,
                density=density,
                bins=bins,
                labels=labels,
                colours=colours,
                hatches=hatches,
                preds=preds,
                **kwargs,
            )
        if data.shape[1] == 2:
            return _PlotClusters2D(
                data,
                targets,
                density=density,
                bins=bins,
                labels=labels,
                colours=colours,
                hatches=hatches,
                preds=preds,
                **kwargs,
            )
        if data.shape[1] == 3 and plot_3d:
            return _PlotClusters3D(
                data,
                targets,
                density=density,
                bins=bins,
                labels=labels,
                colours=colours,
                hatches=hatches,
                preds=preds,
                **kwargs,
            )
        return _PlotClustersND(
            data,
            targets,
            density=density,
            bins=bins,
            labels=labels,
            colours=colours,
            hatches=hatches,
            preds=preds,
            **kwargs,
        )
