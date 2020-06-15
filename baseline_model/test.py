from pathlib import Path

from datasets.constants import DatasetName, DatasetType
from datasets.tf_record_utils import tf_record_dataset, transform_for_signum_model
from training.utils import model_path, load_model


def _test_dataset():
    test_dataset = tf_record_dataset(DatasetName.SIGNUM, DatasetType.TEST)
    test_dataset = test_dataset.batch(32)
    test_dataset = test_dataset.map(transform_for_signum_model)
    return test_dataset.prefetch(2)


def test():
    test_dataset = _test_dataset()
    path = model_path(Path(), TRAINING_RUN)
    model = load_model(path)
    return model.evaluate(test_dataset)


TRAINING_RUN = '20200613_234623'  # 16 signers yielding 7200 training examples
# TRAINING_RUN = '20200614_090325'  # 8 signers yielding 3600 training examples
# TRAINING_RUN = '20200614_202553'  # 4 signers yielding 1800 training examples

if __name__ == '__main__':
    loss, accuracy = test()
    print(f'Loss: {loss}, accuracy: {accuracy}')
