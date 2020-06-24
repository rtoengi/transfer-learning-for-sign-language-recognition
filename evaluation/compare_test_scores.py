import itertools

import pandas as pd

import baseline_model as baseline_model_package
import finetuned_model as finetuned_model_package
from core.constants import Metric
from core.utils import package_path, load_dataframe
from plotting.constants import COMPARE_TEST_SCORES_COLUMNS
from plotting.plot import compare_test_scores_plot
from testing.utils import scores_file_path


def dataframe():
    path = scores_file_path(package_path(baseline_model_package))
    baseline_df = load_dataframe(path)
    path = scores_file_path(package_path(finetuned_model_package))
    finetuned_df = load_dataframe(path)
    return pd.concat([baseline_df[Metric.LOSS.value], finetuned_df[Metric.LOSS.value],
                      baseline_df[Metric.ACCURACY.value], finetuned_df[Metric.ACCURACY.value]], axis=1,
                     keys=list(itertools.chain(*COMPARE_TEST_SCORES_COLUMNS)))


def compare_test_scores():
    df = dataframe()
    compare_test_scores_plot(df)


if __name__ == '__main__':
    compare_test_scores()
