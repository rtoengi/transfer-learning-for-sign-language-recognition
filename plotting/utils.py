from pandas import DataFrame


def _start_index_from_one(df: DataFrame):
    """Increments the index of the DataFrame by one.

    Arguments:
        df: A DataFrame.
    """
    df.index += 1
