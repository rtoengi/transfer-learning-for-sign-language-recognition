import pandas as pd
from pandas import DataFrame

from training.constants import _HISTORY_FILE_NAME


def save_dataframe(dataframe: DataFrame, path):
    """Saves a DataFrame as a pickled object.

    Arguments:
        dataframe: The DataFrame to be saved.
        path: A string representing the path where the DataFrame will be saved.
    """
    dataframe.to_pickle(path)


def load_dataframe(path):
    """Loads a DataFrame from a stored pickled object.

    Arguments:
        path: A string representing the path where the DataFrame will be loaded from.

    Returns:
        The loaded DataFrame.
    """
    return pd.read_pickle(path)


def _history_path(path):
    """Returns the path of the history file.

    Arguments:
        path: A string representing the path to the directory where the history file is located.

    Returns:
        A string representing the path of the history file.
    """
    return f'{path}/{_HISTORY_FILE_NAME}'


def save_history(history, path):
    """Saves the history of a training run.

    Arguments:
        history: The history of a training run to be saved.
        path: A string representing the path where the history will be saved.
    """
    save_dataframe(pd.DataFrame(history), _history_path(path))
