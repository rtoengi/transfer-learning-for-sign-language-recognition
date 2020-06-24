import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame

from plotting.constants import _LABELS, _LOSS_ACCURACY_COLUMNS, COMPARE_TRAINING_COLUMNS, COMPARE_TEST_SCORES_COLUMNS
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
        sns.lineplot(data=df[COMPARE_TRAINING_COLUMNS[i]], markers=['o', 'o'], ax=axes[i])
        axes[i].set(title=f"{_LABELS['train_validation'][i]} dataset", xlabel='Epoch', ylabel='Accuracy')
        axes[i].legend(title='Model', labels=['Baseline', 'Fine-tuned'])
    fig.show()


def compare_test_scores_plot(df: DataFrame):
    """Plots the losses and accuracies of a baseline and a fine-tuned model for each of the different dataset sizes.

    Arguments:
        df: The DataFrame of the merged test scores files of a baseline and a fine-tuned model.
    """
    df.index = _LABELS['dataset_sizes']
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 4.8))
    for i in range(2):
        sns.lineplot(data=df[COMPARE_TEST_SCORES_COLUMNS[i]], markers=['o', 'o'], ax=axes[i])
        axes[i].set(xticks=df.index, title=f"{_LABELS['loss_accuracy'][i]} comparison", xlabel='Dataset size',
                    ylabel=_LABELS['loss_accuracy'][i])
        axes[i].legend(title='Model', labels=['Baseline', 'Fine-tuned'])
    fig.show()


def test_scores_improvement_plot(df: DataFrame):
    """Plots the improvements of the losses and accuracies for each of the different dataset sizes.

    Arguments:
        df: The DataFrame of the merged test scores files of a baseline and a fine-tuned model.
    """
    df.columns = _LABELS['dataset_sizes']
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 4.8))
    for i in range(2):
        sns.barplot(data=df.iloc[[i]], ax=axes[i])
        yticks = np.linspace(-1.2, 0, 7) if i == 0 else np.linspace(0, 0.22, 12)
        axes[i].set(yticks=yticks, title=f"{_LABELS['loss_accuracy'][i]} improvement", xlabel='Dataset size',
                    ylabel=_LABELS['loss_accuracy'][i])
    fig.show()
