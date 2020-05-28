from datasets.constants import DatasetName, DatasetType
from datasets.inspection.display import _play_frames
from datasets.tf_record_utils import tf_record_dataset, transform_for_inspection


def inspect_dataset(dataset_name: DatasetName, dataset_type: DatasetType,
                    inspect_fn, batch_size=1, skip_count=0, take_count=1):
    """Inspects a given number of dataset examples according to a passed inspection function.

    Arguments:
        dataset_name: The name of the dataset to inspect (one of `DatasetName`).
        dataset_type: The type of the dataset to inspect (one of `DatasetType`).
        inspect_fn: The function to apply to a sequence of frames.
        batch_size: The number of consecutive dataset examples to combine into a batch.
        skip_count: The number of dataset examples to skip.
        take_count: The maximum number of dataset examples to fetch.
    """
    dataset = tf_record_dataset(dataset_name, dataset_type)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(transform_for_inspection)
    if skip_count:
        dataset = dataset.skip(skip_count)
    for frames_batch, labels, signers in dataset.take(take_count):
        frames_batch = frames_batch.numpy()
        for frames in frames_batch:
            inspect_fn(frames)


if __name__ == '__main__':
    inspect_dataset(
        dataset_name=DatasetName.MSASL,
        dataset_type=DatasetType.TRAIN,
        inspect_fn=_play_frames,
        batch_size=1,
        skip_count=0,
        take_count=1
    )
