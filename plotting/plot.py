import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from plotting.constants import _LABELS, _LOSS_ACCURACY_COLUMNS, BASELINE_FINETUNED_COLUMNS
from plotting.utils import _start_index_from_one

sns.set()


def loss_accuracy_plot(df: DataFrame):
    """Plots the learning curves of the losses and accuracies for each of the train and validation datasets.

    Arguments:
        df: The DataFrame of a history file.
    """
    _start_index_from_one(df)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 4.8))
    for i in range(2):
        sns.lineplot(data=df[_LOSS_ACCURACY_COLUMNS[i]], markers=['o', 'o'], ax=axes[i])
        axes[i].set(title=f"{_LABELS['loss_accuracy'][i]} learning curves", xlabel='Epoch',
                    ylabel=_LABELS['loss_accuracy'][i])
        axes[i].legend(title='Dataset', labels=['Train', 'Validation'])
    fig.show()


def loss_accuracy_without_validation_plot(df: DataFrame):
    """Plots the learning curves of the loss and accuracy for the train dataset.

    Arguments:
        df: The DataFrame of a history file.
    """
    _start_index_from_one(df)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 4.8))
    for i in range(2):
        sns.lineplot(data=df[_LOSS_ACCURACY_COLUMNS[i][0]], marker='o', ax=axes[i])
        axes[i].set(xticks=df.index, title=f"{_LABELS['loss_accuracy'][i]} learning curve", xlabel='Epoch',
                    ylabel=_LABELS['loss_accuracy'][i])
        axes[i].legend(title='Dataset', labels=['Train'])
    fig.show()


def compare_training_plot(df: DataFrame):
    """Plots the learning curves of a baseline and a fine-tuned model for each of the train and validation datasets.

    Arguments:
        df: The DataFrame of the merged history files of a baseline and a fine-tuned model.
    """
    _start_index_from_one(df)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 4.8))
    for i in range(2):
        sns.lineplot(data=df[BASELINE_FINETUNED_COLUMNS[i]], markers=['o', 'o'], ax=axes[i])
        axes[i].set(title=f"{_LABELS['train_validation'][i]} dataset", xlabel='Epoch', ylabel='Accuracy')
        axes[i].legend(title='Model', labels=['Baseline', 'Fine-tuned'])
    fig.show()
