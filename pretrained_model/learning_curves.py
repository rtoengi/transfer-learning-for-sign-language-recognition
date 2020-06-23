from pathlib import Path

from core.utils import load_dataframe
from plotting.plot import loss_accuracy_without_validation_plot
from training.utils import history_path

TRAINING_RUN = '20200612_235400'


def plot_learning_curves():
    path = history_path(Path(), TRAINING_RUN)
    df = load_dataframe(path)
    loss_accuracy_without_validation_plot(df)


if __name__ == '__main__':
    plot_learning_curves()
