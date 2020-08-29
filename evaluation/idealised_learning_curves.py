import numpy as np
from pandas import DataFrame
from scipy.interpolate import UnivariateSpline

from core.constants import Metric
from plotting.plot import idealised_learning_curves_plot


def _splines():
    spline = UnivariateSpline([1, 10, 20, 30, 40], [4, 1.5, 0.5, 0.275, 0.225], k=4)
    val_spline = UnivariateSpline([1, 10, 20, 30, 40], [4.5, 2, 1.4, 1.675, 2.25], k=4)
    xs = np.linspace(1, 40, 40)
    return spline(xs), val_spline(xs)


def idealised_learning_curves():
    spline, val_spline = _splines()
    df = DataFrame({
        Metric.LOSS.value: spline,
        Metric.VAL_LOSS.value: val_spline
    })
    idealised_learning_curves_plot(df)


if __name__ == '__main__':
    idealised_learning_curves()
