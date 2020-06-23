from pathlib import Path

import pandas as pd
from pandas import DataFrame


def package_path(package):
    """Returns the location of the passed package.

    Arguments:
        package: A package object.

    Returns:
        An absolute Path object pointing to the package's location.
    """
    return Path(package.__path__[0])


def save_dataframe(df: DataFrame, path):
    """Saves a DataFrame as a pickled object.

    Arguments:
        df: The DataFrame to be saved.
        path: A string representing the path where the DataFrame will be saved.
    """
    df.to_pickle(path)


def load_dataframe(path):
    """Loads a DataFrame from a saved pickled object.

    Arguments:
        path: A string representing the path where the DataFrame will be loaded from.

    Returns:
        The loaded DataFrame.
    """
    return pd.read_pickle(path)
