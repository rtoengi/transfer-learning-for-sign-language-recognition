import json

from datasets.constants import DatasetName, DatasetType
from datasets.msasl.constants import _MSASL_FILTERED_SPECS_DIR
from datasets.tf_record_utils import _dataset_counts
from datasets.utils import _display_dataset_counts


def display_dataset_example_spec():
    """Displays a specification entry of the `MS-ASL` dataset."""
    with open(f'{_MSASL_FILTERED_SPECS_DIR}/{DatasetType.TRAIN.value}.json', 'r') as file:
        dataset = json.load(file)
    print('MSASL dataset example spec')
    print('=' * 27)
    print(json.dumps(dataset[0], indent=4))


if __name__ == '__main__':
    _display_dataset_counts(DatasetName.MSASL, _dataset_counts(DatasetName.MSASL))
    print()
    display_dataset_example_spec()
