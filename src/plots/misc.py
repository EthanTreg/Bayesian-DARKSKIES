"""
Miscellaneous plotting classes
"""
from typing import Any

import matplotlib as mpl
import numpy as np
from numpy import ndarray
from matplotlib.axes import Axes
from netloader.utils.utils import label_change

from src.plots import utils
from src.plots.base import BasePlot
from src.plots.utils import subplot_grid


class PlotConfusion(BasePlot):
    """
    Plots the confusion matrix between targets and predictions

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
            labels: list[str],
            preds: ndarray,
            targets: ndarray,
            fig_size: tuple[int, int] = utils.HI_RES_SQUARE,
            **kwargs: Any):
        """
        Parameters
        ----------
        labels : list[str]
            Labels for each class
        preds : (N) ndarray
            Predicted value for N data points
        targets : (N) ndarray
            Target values for N data points
        fig_size : tuple[int, int], default = HI_RES_SQUARE
            Size of the figure

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._labels: list[str]
        self._data: ndarray
        self._targets: ndarray = targets
        self._classes: ndarray = np.unique(self._targets)
        super().__init__(
            preds,
            labels=labels,
            fig_size=fig_size,
            **kwargs,
        )

    def _post_init(self) -> None:
        self._default_name = 'confusion'

    def _plot_data(self) -> None:
        assert isinstance(self.axes, Axes)
        i: int
        j: int
        value: float
        class_: float
        colour: str
        idxs: ndarray
        counts: ndarray
        matrix_row: ndarray
        class_preds: ndarray
        matrix: ndarray = np.zeros((len(self._classes), len(self._classes)))

        # Generate confusion matrix
        for matrix_row, class_ in zip(matrix, self._classes):
            idxs = self._targets == class_
            class_preds, counts = np.unique(self._data[idxs], return_counts=True)
            class_preds = label_change(class_preds, self._classes)
            matrix_row[class_preds] = counts / np.count_nonzero(idxs) * 100

        self.plots.append(self.axes.imshow(matrix, cmap='Blues'))
        self.axes.set_xticks(np.arange(len(self._labels)), self._labels, fontsize=utils.MINOR)
        self.axes.set_yticks(
            np.arange(len(self._labels)),
            self._labels,
            rotation=90,
            va='center',
            fontsize=utils.MINOR,
        )
        self.axes.set_xlabel('Predictions', fontsize=utils.MAJOR)
        self.axes.set_ylabel('Targets', fontsize=utils.MAJOR)

        for (i, j), value in zip(np.ndindex(matrix.shape), matrix.flatten()):
            colour = 'w' if value > 50 else 'k'
            self.axes.text(
                j, i, f'{value:.1f}',
                ha='center',
                va='center',
                color=colour,
                fontsize=utils.MINOR,
            )

    def create_legend(self, *_, **__) -> None:
        """
        Confusion matrix should not have a legend
        """
        return


class PlotDistributions(BasePlot):
    """
    Plots the distributions of data points

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
            data: list[ndarray] | ndarray,
            log: bool = False,
            norm: bool = False,
            y_axes: bool = True,
            density: bool = False,
            bins: int = 100,
            num_plots: int = 12,
            x_label: str = '',
            text_loc: str = 'upper right',
            labels: str | tuple[str, str] = '',
            hatches: str | tuple[str, str] = '',
            texts: list[str] | None = None,
            titles: list[str] | None = None,
            data_twin: list[ndarray] | ndarray | None = None,
            **kwargs: Any):
        """
        Parameters
        ----------
        data : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B distributions with N data points
        log : bool, default = False
            If the x-axis should be logarithmic
        norm : bool, default = False
            If the distributions should be normalised to a maximum height of 1
        y_axes : bool, default = True
            If the y-axis should have ticks
        density : bool, default = False
            If the distributions should be histograms or density plots
        bins : int, default = 200
            Number of bins for histograms or density interpolation
        num_plots : int, default = 12
            Maximum number of distributions to plot
        x_label : str, default = ''
            X-axis label
        text_loc : str, default = 'upper right'
            Location of the text
        labels : str | tuple[str, str], default = ''
            Labels for the data and twin data
        hatches : str | tuple[str, str], default = ''
            Hatching pattern for the distributions and twin data
        texts : list[str] | None, default = None
            Texts to be displayed on the distributions
        titles : list[str] | None, default = None
            Titles for the distributions
        data_twin : list[(M) ndarray] | (M) ndarray | (B,M) ndarray | None, default = None
            B twin distributions with M data points to plot on each distribution

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._log: bool = log
        self._norm: bool
        self._y_axes: bool = y_axes
        self._num_plots: int
        self._text_loc: str = text_loc
        self._hatches: str | tuple[str, str] | None = hatches
        self._titles: list[str] | None = titles
        self._texts: list[str]
        self._data: list[ndarray] | ndarray = [data] \
            if isinstance(data, ndarray) and np.ndim(data) < 2 else data
        self._data_twin: list[ndarray] | list[None] | ndarray = [None] * len(self._data) \
            if data_twin is None else data_twin

        self._norm = True if any(datum.size == 1 for datum in self._data) else norm
        self._num_plots = min(num_plots, len(self._data))
        self._texts = texts or [''] * len(self._data)
        super().__init__(
            self._data,
            density=density,
            bins=bins,
            x_label=x_label,
            labels=list(labels) if isinstance(labels, tuple) else [labels],
            **kwargs,
        )

    def _axes_init(self) -> None:
        self.subplots(subplot_grid(self._num_plots), titles=self._titles)

    def _post_init(self) -> None:
        self._default_name = 'distributions'

    def _plot_data(self) -> None:
        assert isinstance(self.axes, dict)
        text: str
        range_: tuple[float, float]
        data: ndarray
        data_twin: ndarray
        axis: Axes

        for text, data, data_twin, axis in zip(
                self._texts,
                self._data,
                self._data_twin,
                self.axes.values()):
            range_ = (
                    min(np.min(data), np.min(data_twin)),
                    max(np.max(data), np.max(data_twin)),
                )
            axis.set_xscale('log' if self._log else 'linear')
            self.plot_histogram(
                self._colours[0],
                data,
                axis,
                log=self._log,
                norm=self._norm,
                hatch=self._hatches[0] if isinstance(self._hatches, tuple) else self._hatches,
                range_=range_,
            )
            self.plot_histogram(
                self._colours[1],
                data_twin,
                axis,
                log=self._log,
                norm=self._norm,
                hatch=self._hatches[1] if isinstance(self._hatches, tuple) else self._hatches,
                range_=(
                    min(np.min(data), np.min(data_twin)),
                    max(np.max(data), np.max(data_twin)),
                ),
            )

            if not self._y_axes:
                axis.tick_params(labelleft=False, left=False)

            if text:
                axis.add_artist(mpl.offsetbox.AnchoredText(
                    text,
                    loc=self._text_loc,
                    prop={'fontsize': utils.MINOR}
                ))


class PlotGaussianPreds(BasePlot):
    """
    Plots Gaussian predictions and their uncertainties

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
            preds: list[ndarray] | ndarray,
            targets: list[ndarray] | ndarray,
            labels: list[str] | None = None,
            uncertainties: list[ndarray] | ndarray | None = None,
            **kwargs: Any):
        """
        Parameters
        ----------
        preds : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B sets of N Gaussian predictions
        targets : list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B sets of N targets
        labels : list[str] | None = None
            Labels for each set of Gaussian predictions
        uncertainties : list[(N) ndarray] | (N) ndarray | (B,N) ndarray | None, default = None
            B sets of N uncertainties

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._data: list[ndarray] | ndarray = [preds] if np.ndim(preds) < 2 else preds
        self._targets: list[ndarray] | ndarray = [targets] * len(preds) \
            if np.ndim(targets) < 2 else targets
        self._uncertainties: list[ndarray] | ndarray | None = [uncertainties] * len(preds) \
            if np.ndim(uncertainties) < 2 else uncertainties
        self._major_axes: list[Axes] = []
        super().__init__([preds] if np.ndim(preds) == 1 else preds, labels=labels, **kwargs)

    def _post_init(self) -> None:
        self._default_name = 'gaussian_preds'

    def _plot_data(self) -> None:
        colour: str
        data: ndarray
        target: ndarray
        uncertainty: ndarray

        for colour, data, target, uncertainty in zip(
                self._colours,
                self._data,
                self._targets,
                self._uncertainties):
            self._major_axes.append(self.plot_reconstruction(
                colour,
                target,
                data,
                target,
                self.axes,
                uncertainty=uncertainty,
                major_axis=self._major_axes[-1] if self._major_axes else None,
            ))
            self.plots.append(self._major_axes[-1].plot(
                *([np.min(target), np.max(target)] * 2),
                color='k',
            )[0])


class PlotImages(BasePlot):
    """
    Plots images as a grid

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
            num_plots: int = 12,
            cmaps: str | list[str] | None = None,
            titles: list[str] | None = None,
            fig_size: tuple[int, int] = utils.HI_RES,
            ranges: tuple[float, float] | list[tuple[float, float]] | ndarray | None = None,
            **kwargs: Any):
        """
        Parameters
        ----------
        data : (B,W,H) ndarray
            B image data of width W and height H
        num_plots : int, default = 12
            Maximum number of images to plot
        cmaps : str | list[str] | None, default = 'hot'
            Colour map(s) for each image
        titles: list[str] | None, default = None
            Titles for each image
        fig_size : tuple[int, int], default = HI_RES
            Size of the figure
        ranges : tuple[float, float] | list[tuple[float, float]] | ndarray | None, default = None
            Lower and upper bounds for each image, default is minimum and maximum for each image

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._num_plots: int = min(num_plots, len(data))
        self._cmaps: list[str]
        self._ranges: list[tuple[float, float]] | ndarray
        self._data: ndarray

        if isinstance(cmaps, str):
            self._cmaps = [cmaps] * self._num_plots
        else:
            self._cmaps = cmaps or ['hot'] * self._num_plots

        if ranges is None:
            self._ranges = [(np.min(data), np.max(data))] * self._num_plots
        elif np.ndim(ranges) < 2:
            self._ranges = [ranges] * self._num_plots
        else:
            self._ranges = ranges

        super().__init__(data, labels=titles, fig_size=fig_size, **kwargs)

    def _axes_init(self) -> None:
        self.subplots(subplot_grid(self._num_plots), titles=self._labels)

    def _post_init(self) -> None:
        self._default_name = 'images'

    def _plot_data(self) -> None:
        assert isinstance(self.axes, dict)
        cmap: str
        range_: tuple[float, float] | ndarray
        data: ndarray
        axis: Axes

        for cmap, range_, data, axis in zip(
                self._cmaps,
                self._ranges,
                self._data,
                self.axes.values()):
            self.plots.append(axis.imshow(data, cmap=cmap, vmin=range_[0], vmax=range_[1]))
            axis.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)

    def create_legend(self, *_, **__) -> None:
        """
        Confusion matrix should not have a legend
        """
        return


class PlotParamComparison(BasePlot):
    """
    Plots a comparison between y-data and x-data

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
            x_data: ndarray,
            y_data: ndarray,
            **kwargs: Any):
        """
        Parameters
        ----------
        x_data : (N) ndarray
            N x-axis data points
        y_data : (N) ndarray
            N y-axis data points

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._data: ndarray
        self._x_data: ndarray = x_data
        super().__init__(y_data, **kwargs)

    def _post_init(self) -> None:
        self._default_name = 'comparison'

    def _plot_data(self) -> None:
        assert isinstance(self.axes, Axes)
        range_: tuple[float, float] = np.min(self._x_data), np.max(self._x_data)

        self.plots.append(self.axes.scatter(
            self._x_data,
            self._data,
            alpha=self._alpha,
            color=self._colours[0],
        ))
        self.plots.append(self.axes.plot(range_, range_, color='k')[0])
        self.axes.xaxis.get_offset_text().set_visible(False)
        self.axes.yaxis.get_offset_text().set_size(utils.MINOR)
        self.axes.text(
            0.1,
            0.9,
            rf'$\chi^2_\nu=${np.mean((self._data - self._x_data) ** 2 / self._x_data):.2f}',
            fontsize=utils.MINOR,
            transform=self.axes.transAxes,
        )


class PlotParamPairs(BasePlot):
    """
    Plots a pair plot to compare the distributions and comparisons between parameters

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
            data: list[ndarray] | ndarray,
            density: bool = False,
            labels: list[str] | None = None,
            fig_size: tuple[int, int] = utils.HI_RES,
            **kwargs: Any):
        """
        Parameters
        ----------
        data : list[(N,L) ndarray] | (N,L) ndarray | (B,N,L) ndarray
            B sets of N data points for L parameters
        density : bool, default = False
            If the data should be plotted as contours and density plots or histograms
        labels : list[str] | None, default = None
            Labels for each set of parameter comparisons
        fig_size : tuple[int, int], default = HI_RES
            Size of the figure

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        pad: float = 0.05
        self._labels: list[str]
        self._data: list[ndarray] | ndarray = [data] if np.ndim(data) < 3 else data
        self._ranges: ndarray = np.stack((
            np.min([np.min(datum - np.abs(pad * datum)) for datum in self._data]),
            np.max([np.max(datum + np.abs(pad * datum)) for datum in self._data]),
        ), axis=1)
        super().__init__(self._data, density=density, labels=labels, fig_size=fig_size, **kwargs)

    def _axes_init(self) -> None:
        self.subplots((len(self._data),) * 2, sharex='col')

    def _post_init(self) -> None:
        self._default_name = 'param_pairs'

    def _plot_data(self) -> None:
        for colour, data in zip(self._colours, self._data):
            self.plot_param_pairs(colour, data, ranges=self._ranges)


class PlotPerformance(BasePlot):
    """
    Plots the performance over epochs of training

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
            data: list[float] | list[ndarray] | ndarray,
            log: bool = True,
            x_label: str = '',
            y_label: str = '',
            labels: list[str] | None = None,
            **kwargs: Any):
        """
        Parameters
        ----------
        data : list[float] | list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B sets of N performance metrics over epochs
        log : bool, default = True
            If the y-axis should be logarithmic
        x_label : str, default = ''
            X-axis label
        y_label : str, default = ''
            Y-axis label
        labels : list[str] | None, default = None
            Labels for each set of performance metrics

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._log = log
        self._data: list[list[float] | ndarray] = [data] if np.ndim(data) < 2 else data
        super().__init__(self._data, x_label=x_label, y_label=y_label, labels=labels, **kwargs)

    def _post_init(self) -> None:
        self._default_name = 'performance'

    def _plot_data(self) -> None:
        assert isinstance(self.axes, Axes)
        self.axes.set_yscale('log' if self._log else 'linear')

        if self._labels and self._labels[0]:
            self.axes.text(
                0.7, 0.75,
                '\n'.join(
                    (f'Final {label}: {data[-1]:.3e}'
                     for label, data in zip(self._labels, self._data)),
                ),
                fontsize=utils.MINOR,
                transform=self.axes.transAxes
            )

        for colour, data in zip(self._colours, self._data):
            self.plots.append(self.axes.plot(data, color=colour)[0])


class PlotSaliency(PlotImages):
    """
    Plots the saliency and input for multiple saliency maps

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
            saliency: ndarray,
            num_plots: int = 12,
            fig_size: tuple[int, int] = utils.HI_RES,
            **kwargs: Any):
        """
        Parameters
        ----------
        data : (W,H) ndarray
            Original input data of width W and height H
        saliency : (B,W,H) ndarray
            B sets of saliency maps of width W and height H
        num_plots : int, default = 12
            Maximum number of saliencies to plot
        fig_size : tuple[int, int], default = HI_RES
            Size of the figure

        **kwargs
            Optional keyword arguments to pass to PlotImages
        """
        super().__init__(
            np.concat((data[np.newaxis], saliency)),
            num_plots=num_plots,
            cmaps=['hot'] + ['twilight'] * len(saliency),
            titles=['Input'] + [f'Dim {i + 1}' for i in range(len(saliency))],
            ranges=np.concat((
                [[0, np.max(data)]],
                np.min(saliency, axis=(1, 2)),
                np.max(saliency, axis=(1, 2)),
            )),
            fig_size=fig_size,
            **kwargs,
        )

    def _post_init(self) -> None:
        self._default_name = 'saliency'
