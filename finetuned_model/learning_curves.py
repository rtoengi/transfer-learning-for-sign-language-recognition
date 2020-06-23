from pathlib import Path

from finetuned_model.constants import TRAINING_RUNS
from plotting.plot import loss_accuracy_plot
from training.utils import history_path, load_dataframe


def plot_learning_curves():
    for training_run in TRAINING_RUNS:
        path = history_path(Path(), training_run)
        df = load_dataframe(path)
        loss_accuracy_plot(df)


if __name__ == '__main__':
    plot_learning_curves()
