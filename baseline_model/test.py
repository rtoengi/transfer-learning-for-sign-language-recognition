from pathlib import Path

from baseline_model.constants import TRAINING_RUNS
from datasets.constants import DatasetName, DatasetType
from datasets.signum.constants import TrainDatasetSize
from datasets.tf_record_utils import tf_record_dataset, transform_for_signum_model
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
    losses, accuracies = test()
    # loss, accuracy = test()
    # print(f"Test scores for the baseline model trained on {train_dataset_size.value['n_examples']} examples of "
    #       f"{train_dataset_size.value['signers']} signers:")
    # print(f'Loss: {loss}, accuracy: {accuracy}')
