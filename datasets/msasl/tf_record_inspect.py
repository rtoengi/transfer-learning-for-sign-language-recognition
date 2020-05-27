import cv2

from datasets.constants import DatasetName, DatasetType
from datasets.tf_record_utils import tf_record_dataset, transform_for_inspection


def _display_images(images):
    for image in images:
        cv2.imshow('Frame', image)
        cv2.waitKey(0)


def inspect_dataset():
    dataset = tf_record_dataset(DatasetName.MSASL, DatasetType.TRAIN)
    dataset = dataset.batch(1)
    dataset = dataset.map(transform_for_inspection)
    for frames, labels, signers in dataset.take(1):
        frames = frames.numpy()
        _display_images(frames[0][0:1])


if __name__ == '__main__':
    inspect_dataset()
