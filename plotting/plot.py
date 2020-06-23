import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from plotting.constants import _LABELS, _LOSS_ACCURACY_COLUMNS
from plotting.utils import _start_index_from_one

sns.set()


def loss_accuracy_learning_curves(df: DataFrame, model_type, dataset_size):
    """Plots the learning curves of the losses and accuracies for each of the train and validation datasets.

    Arguments:
        df: The DataFrame of a history file.
        model_type: The label of the type of model.
        dataset_size: The label of the size of the dataset.
    """
    _start_index_from_one(df)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 4.8))
    fig.suptitle(_LABELS['learning_curves'].format(model_type=model_type, dataset_size=dataset_size))
    for i in range(2):
        sns.lineplot(data=df[_LOSS_ACCURACY_COLUMNS[i]], markers=['o', 'o'], ax=axes[i])
        axes[i].set(title=' ', xlabel='Epoch', ylabel=_LABELS['loss_accuracy'][i])
        axes[i].legend(title='Dataset', labels=('Train', 'Validation'))
    fig.show()
