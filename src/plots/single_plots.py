"""
Plots that use a single axis
"""
import logging
from typing import Any

import numpy as np
from numpy import ndarray
from matplotlib.axes import Axes

from src.plots import utils
from src.plots.base import BasePlot


class BasePlots(BasePlot):
    """
    Base class to plot multiple sets of data on the same axes

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
            x_data: list[ndarray] | ndarray,
            y_data: list[ndarray] | ndarray,
            log_x: bool = False,
            log_y: bool = False,
            error_region: bool = False,
            x_label: str = '',
            y_label: str = '',
            labels: list[str] | None = None,
            x_error: list[ndarray] | ndarray | None = None,
            y_error: list[ndarray] | ndarray | None = None,
            **kwargs: Any):
        """
        Parameters
        ----------
        data : list[float] | list[(N) ndarray] | (N) ndarray | (B,N) ndarray
            B sets of N values to plot
        log_x : bool, default = False
            If the x-axis should be logarithmic
        log_y : bool, default = False
            If the y-axis should be logarithmic
        x_label : str, default = ''
            X-axis label
        y_label : str, default = ''
            Y-axis label
        labels : list[str] | None, default = None
            Labels for each set of data

        **kwargs
            Optional keyword arguments to pass to BasePlot
        """
        self._log_x: bool = log_x
        self._log_y: bool = log_y
        self._error_region: bool = error_region
        self._data: list[ndarray] = [x_data] if np.ndim(x_data[0]) < 1 else x_data
        self._y_data: list[ndarray] = [y_data] if np.ndim(y_data[0]) < 1 else y_data
        self._x_error: list[ndarray] | list[None]
        self._y_error: list[ndarray] | list[None]

        if len(self._data) == 1:
            self._data *= len(self._y_data)

        if len(self._y_data) == 1:
            self._y_data *= len(self._data)

        if x_error is None:
            self._x_error = [None] * len(self._data)
        else:
            self._x_error = [x_error] if np.ndim(x_error[0]) < 1 else x_error

        if y_error is None:
            self._y_error = [None] * len(self._y_data)
        else:
            self._y_error = [y_error] if np.ndim(y_error[0]) < 1 else y_error

        if self._error_region and self._x_error[0]:
            logging.getLogger(__name__).warning('X-error is ignored if error_region is True')

        super().__init__(self._data, x_label=x_label, y_label=y_label, labels=labels, **kwargs)

    def _post_init(self) -> None:
        self._default_name = 'plots'

    def _plot_data(self) -> None:
        assert isinstance(self.axes, Axes)
        self.axes.set_xscale('log' if self._log_x else 'linear')
        self.axes.set_yscale('log' if self._log_y else 'linear')

        for (colour,
             x_data,
             y_data,
             x_error,
             y_error) in zip(self._colours, self._data, self._y_data, self._x_error, self._y_error):
            self.plots.append(self.axes.plot(x_data, y_data, color=colour)[0])

            if self._error_region:
                self.plots.append(self.axes.fill_between(
                    x_data,
                    y_data + y_error,
                    y_data - y_error,
                    color=colour,
                    alpha=self._alpha_2d
                ))
            else:
                self.plots.append(self.axes.errorbar(
                    x_data,
                    y_data,
                    yerr=y_error,
                    xerr=x_error,
                    color=colour,
                )[0])


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


class PlotPerformance(BasePlots):
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
            x_data: list[float] | list[ndarray] | ndarray | None = None,
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
        x_data : list[str] | list[(N) ndarray] | (N) ndarray | (B,N) ndarray | None, default = None
            B sets of N x-axis data points

        **kwargs
            Optional keyword arguments to pass to PlotPlots
        """
        self._y_data: list[ndarray] | ndarray = [data] if np.ndim(data[0]) < 1 else data
        super().__init__(
            x_data or [np.arange(len(datum)) for datum in self._y_data],
            self._y_data,
            log_y=log,
            x_label=x_label,
            y_label=y_label,
            labels=labels,
            **kwargs,
        )

    def _post_init(self) -> None:
        self._default_name = 'performance'

    def _plot_data(self) -> None:
        assert isinstance(self.axes, Axes)
        super()._plot_data()

        if self._labels and self._labels[0]:
            self.axes.text(
                0.7, 0.75,
                '\n'.join(
                    (f'Final {label}: {data[-1]:.3e}'
                     for label, data in zip(self._labels, self._y_data)),
                ),
                fontsize=utils.MINOR,
                transform=self.axes.transAxes
            )


PlotPlots = BasePlots
