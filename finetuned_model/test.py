from pathlib import Path

from datasets.constants import DatasetName, DatasetType
from datasets.tf_record_utils import tf_record_dataset, transform_for_signum_model
from finetuned_model.constants import TRAINING_RUNS
from testing.utils import save_scores, scores_file_exists, display_scores
from training.utils import model_path, load_model


def _test_dataset():
    test_dataset = tf_record_dataset(DatasetName.SIGNUM, DatasetType.TEST)
    test_dataset = test_dataset.batch(32)
    test_dataset = test_dataset.map(transform_for_signum_model)
    return test_dataset.prefetch(2)


def test():
    losses, accuracies = [], []
    test_dataset = _test_dataset()
    for training_run in TRAINING_RUNS:
        path = model_path(Path(), training_run)
        model = load_model(path)
        loss, accuracy = model.evaluate(test_dataset)
        losses.append(loss)
        accuracies.append(accuracy)
    return losses, accuracies


if __name__ == '__main__':
    if not scores_file_exists(Path()):
        losses, accuracies = test()
        save_scores(losses, accuracies, Path())
    display_scores(Path())
