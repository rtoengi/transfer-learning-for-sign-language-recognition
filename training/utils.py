from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from training.constants import _TRAINING_RUNS_DIR, _HISTORY_FILE_NAME


def create_training_runs_dir(base_path: Path):
    """Creates the directory where the training runs will be stored.

    Arguments:
        base_path: A Path object pointing to the base directory where the training runs will be stored.

    Returns:
        A Path object pointing to the directory of the training runs.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = (base_path / _TRAINING_RUNS_DIR / timestamp).absolute()
    path.mkdir(parents=True)
    return path


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


def _history_path(path: Path):
    """Returns the path of the history file.

    Arguments:
        path: A Path object pointing to the directory where the history file will be stored.

    Returns:
        A Path object of the history file.
    """
    return path / _HISTORY_FILE_NAME


def save_history(history, path: Path):
    """Saves the history of a training run.

    Arguments:
        history: The history of a training run to be saved.
        path: A Path object of the history file.
    """
    save_dataframe(pd.DataFrame(history), _history_path(path))
