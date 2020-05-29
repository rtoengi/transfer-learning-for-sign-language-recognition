import json

from datasets.constants import DatasetType
from datasets.msasl.constants import _MSASL_FILTERED_SPECS_DIR

_LINE_WIDTH = 27


def _dataset_counts():
    """Returns the sizes of the `MS-ASL` train, validation and test datasets.

    Returns:
        A dictionary with an entry of the size for each of the `MS-ASL` train, validation and test datasets.
    """
    counts = {}
    for dataset_type in DatasetType:
        with open(f'{_MSASL_FILTERED_SPECS_DIR}/{dataset_type.value}.json', 'r') as file:
            dataset = json.load(file)
            counts[dataset_type.value] = len(dataset)
    return counts


def display_dataset_counts():
    """Displays the sizes of the `MS-ASL` train, validation and test datasets."""
    counts = _dataset_counts()
    print('MS-ASL dataset counts')
    print('=' * _LINE_WIDTH)
    for dataset_type in DatasetType:
        count = counts[dataset_type.value]
        print(f"{dataset_type.name + ':':13s} {count:5d} ({count / sum(counts.values()) * 100:02.1f}%)")
    print('=' * _LINE_WIDTH)
    print(f"{'Total:':13s} {sum(counts.values()):5d} (100%)")


def display_dataset_example_spec():
    """Displays a specification entry of the `MS-ASL` dataset."""
    with open(f'{_MSASL_FILTERED_SPECS_DIR}/{DatasetType.TRAIN.value}.json', 'r') as file:
        dataset = json.load(file)
    print('MS-ASL dataset example spec')
    print('=' * _LINE_WIDTH)
    print(json.dumps(dataset[0], indent=4))


if __name__ == '__main__':
    display_dataset_counts()
    print()
    display_dataset_example_spec()
