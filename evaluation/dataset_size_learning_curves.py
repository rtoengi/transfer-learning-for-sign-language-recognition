from pandas import DataFrame

from core.constants import Metric
from plotting.plot import dataset_size_learning_curves_plot


def dataset_size_learning_curves():
    df = DataFrame({
        Metric.ACCURACY.value: [0.9989, 0.9953, 1.0],
        Metric.VAL_ACCURACY.value: [0.3217, 0.5128, 0.6689]
    })
    dataset_size_learning_curves_plot(df)


if __name__ == '__main__':
    dataset_size_learning_curves()
