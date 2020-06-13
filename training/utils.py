import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from tensorflow.keras import models
from tensorflow.keras.models import Model

from training.constants import _TRAINING_RUNS_DIR, _SAVED_MODEL_DIR, _HISTORY_FILE_NAME


def create_training_runs_dir(base_path: Path):
    """Creates the directory where the training runs will be saved.

    Arguments:
        base_path: A Path object pointing to the base directory where the training runs will be saved.

    Returns:
        A Path object pointing to the directory of the training runs.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_path = (base_path / _TRAINING_RUNS_DIR / timestamp).absolute()
    base_path.mkdir(parents=True)
    return base_path


def save_dataframe(dataframe: DataFrame, path):
    """Saves a DataFrame as a pickled object.

    Arguments:
        dataframe: The DataFrame to be saved.
        path: A string representing the path where the DataFrame will be saved.
    """
    dataframe.to_pickle(path)


def load_dataframe(path):
    """Loads a DataFrame from a saved pickled object.

    Arguments:
        path: A string representing the path where the DataFrame will be loaded from.

    Returns:
        The loaded DataFrame.
    """
    return pd.read_pickle(path)


def _history_path(path: Path):
    """Returns the path of the history file.

    Arguments:
        path: A Path object pointing to the directory where the history file will be saved.

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


def model_path(base_path: Path, training_run):
    """Returns the path of the model of a given training run.

    Arguments:
        base_path: A Path object pointing to the base directory where the training run in located.
        training_run: The name of the directory of the training run.

    Returns:
        The Path object pointing to the location of the model.
    """
    return base_path / _TRAINING_RUNS_DIR / training_run / _SAVED_MODEL_DIR


def save_model(path: Path, model: Model):
    """Saves a model to disk.

    Arguments:
        path: A Path object pointing to the directory where the model will be saved.
        model: The model to be saved.
    """
    filepath = str(path / _SAVED_MODEL_DIR)
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Saving model to {filepath}')
    model.save(filepath)


def load_model(path: Path):
    """Loads a model from disk.

    Arguments:
        path: A Path object pointing to the directory from where the model will be loaded.

    Returns:
        The model loaded from disk.
    """
    return models.load_model(path)
