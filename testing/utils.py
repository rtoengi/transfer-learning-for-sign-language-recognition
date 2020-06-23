from pathlib import Path

import pandas as pd

from core.constants import Metric
from core.utils import save_dataframe, load_dataframe
from testing.constants import _TESTING_DIR, _SCORES_FILE_NAME


def _create_testing_dir(base_path: Path):
    """Creates the directory where the testing artifacts are located.

    Arguments:
        base_path: A Path object pointing to the base directory where the testing artifacts are located.

    Returns:
        A Path object of the testing directory.
    """
    path = (base_path / _TESTING_DIR).absolute()
    path.mkdir(parents=True)
    return path


def _scores_file_path(base_path: Path):
    """Returns the path of the scores file.

    Arguments:
        base_path: A Path object pointing to the base directory where the testing artifacts are located.

    Returns:
        A Path object of the scores file.
    """
    return base_path / _TESTING_DIR / _SCORES_FILE_NAME


def scores_file_exists(base_path: Path):
    """Checks whether a scores file exists.

    Arguments:
        base_path: A Path object pointing to the base directory where the testing artifacts are located.

    Returns:
        Whether or not a scores file exists.
    """
    path = _scores_file_path(base_path)
    return path.is_file()


def save_scores(losses, accuracies, base_path: Path):
    """Saves a scores file.

    Arguments:
        losses: The list of the loss values.
        accuracies: The list of the accuracy values.
        base_path: A Path object pointing to the base directory where the testing artifacts are located.
    """
    df = pd.DataFrame({
        Metric.LOSS.value: losses,
        Metric.ACCURACY.value: accuracies
    })
    path = _create_testing_dir(base_path)
    save_dataframe(df, path / _SCORES_FILE_NAME)


def display_scores(base_path: Path):
    """Displays a scores file.

    Arguments:
        base_path: A Path object pointing to the base directory where the testing artifacts are located.
    """
    path = _scores_file_path(base_path)
    df = load_dataframe(path)
    print(df)
