from pathlib import Path

from plotting.plot import loss_accuracy_learning_curves
from plotting.utils import training_run_prefix
from training.utils import history_path, load_dataframe

TRAINING_RUNS = [
    'large_dataset/20200616_150747',
    'medium_dataset/20200617_073349',
    'small_dataset/20200617_164302'
]


def plot_learning_curves():
    for training_run in TRAINING_RUNS:
        path = history_path(Path(), training_run)
        df = load_dataframe(path)
        prefix = training_run_prefix(training_run)
        loss_accuracy_learning_curves(df, 'fine-tuned', prefix)


if __name__ == '__main__':
    plot_learning_curves()
