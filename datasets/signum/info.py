from datasets.constants import DatasetName
from datasets.tf_record_utils import _dataset_counts
from datasets.utils import _display_dataset_counts

if __name__ == '__main__':
    _display_dataset_counts(DatasetName.SIGNUM, _dataset_counts(DatasetName.SIGNUM))
