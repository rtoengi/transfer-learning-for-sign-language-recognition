from pathlib import Path

from datasets.constants import DatasetName, DatasetType
from datasets.signum.constants import TrainDatasetSize
from datasets.tf_record_utils import tf_record_dataset, transform_for_signum_model
from training.utils import model_path, load_model

TRAINING_RUNS = {
    TrainDatasetSize.LARGE: '20200616_090434',
    TrainDatasetSize.MEDIUM: '20200616_214425',
    TrainDatasetSize.SMALL: '20200617_143139'
}


def _test_dataset():
    test_dataset = tf_record_dataset(DatasetName.SIGNUM, DatasetType.TEST)
    test_dataset = test_dataset.batch(32)
    test_dataset = test_dataset.map(transform_for_signum_model)
    return test_dataset.prefetch(2)


def test():
    test_dataset = _test_dataset()
    path = model_path(Path(), TRAINING_RUNS[train_dataset_size])
    model = load_model(path)
    return model.evaluate(test_dataset)


# Configure here the size of the train dataset the model has been trained on
train_dataset_size = TrainDatasetSize.LARGE

if __name__ == '__main__':
    loss, accuracy = test()
    print(f"Test scores for the model trained on {train_dataset_size.value['n_examples']} examples of "
          f"{train_dataset_size.value['signers']} signers:")
    print(f'Loss: {loss}, accuracy: {accuracy}')
