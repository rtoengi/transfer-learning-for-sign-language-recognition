import pandas as pd

import baseline_model as baseline_model_package
import finetuned_model as finetuned_model_package
from core.constants import Metric
from core.utils import package_path, load_dataframe
from plotting.plot import test_scores_improvement_plot
from testing.utils import scores_file_path


def dataframe():
    path = scores_file_path(package_path(baseline_model_package))
    baseline_df = load_dataframe(path)
    path = scores_file_path(package_path(finetuned_model_package))
    finetuned_df = load_dataframe(path)
    return pd.concat([finetuned_df[Metric.LOSS.value] - baseline_df[Metric.LOSS.value],
                      finetuned_df[Metric.ACCURACY.value] - baseline_df[Metric.ACCURACY.value]], axis=1,
                     keys=[Metric.LOSS.value, Metric.ACCURACY.value]).transpose()


def test_scores_improvement():
    df = dataframe()
    test_scores_improvement_plot(df)


if __name__ == '__main__':
    test_scores_improvement()
