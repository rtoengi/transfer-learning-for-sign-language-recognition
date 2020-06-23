import itertools

import pandas as pd

import baseline_model as baseline_model_package
import finetuned_model as finetuned_model_package
from baseline_model.constants import TRAINING_RUNS as BASELINE_TRAINING_RUNS
from core.constants import Metric
from core.utils import package_path, load_dataframe
from finetuned_model.constants import TRAINING_RUNS as FINETUNED_TRAINING_RUNS
from plotting.constants import COMPARE_TRAINING_COLUMNS
from plotting.plot import compare_training_plot
from training.utils import history_path


def dataframe(baseline_training_run, finetuned_training_run):
    path = history_path(package_path(baseline_model_package), baseline_training_run)
    baseline_df = load_dataframe(path)
    path = history_path(package_path(finetuned_model_package), finetuned_training_run)
    finetuned_df = load_dataframe(path)
    return pd.concat([baseline_df[Metric.ACCURACY.value], finetuned_df[Metric.ACCURACY.value],
                      baseline_df[Metric.VAL_ACCURACY.value], finetuned_df[Metric.VAL_ACCURACY.value]], axis=1,
                     keys=list(itertools.chain(*COMPARE_TRAINING_COLUMNS)))


def compare_training():
    for i, baseline_training_run in enumerate(BASELINE_TRAINING_RUNS):
        df = dataframe(baseline_training_run, FINETUNED_TRAINING_RUNS[i])
        compare_training_plot(df)


if __name__ == '__main__':
    compare_training()
