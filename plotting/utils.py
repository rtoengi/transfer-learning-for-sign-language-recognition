from pandas import DataFrame


def training_run_prefix(training_run: str):
    """Returns the label of the size of the dataset as indicated by the training run.

    The argument is one of 'large_dataset', 'medium_dataset' or 'small_dataset' with the function returning 'large',
    'medium' or 'small', respectively.

    Arguments:
        training_run: The name of the directory of the training run.

    Returns:
        The label of the size of the dataset as indicated by the training run.
    """
    return training_run[:training_run.index('_')]


def _start_index_from_one(df: DataFrame):
    """Increments the index of the DataFrame by one.

    Arguments:
        df: A DataFrame.
    """
    df.index += 1
