from pathlib import Path

from plotting.plot import loss_accuracy_learning_curves_without_validation
from training.utils import history_path, load_dataframe

TRAINING_RUN = '20200612_235400'


def plot_learning_curves():
    path = history_path(Path(), TRAINING_RUN)
    df = load_dataframe(path)
    loss_accuracy_learning_curves_without_validation(df)


if __name__ == '__main__':
    plot_learning_curves()
