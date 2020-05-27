import os

import cv2
import tensorflow as tf

from datasets.msasl.constants import MSASL_TF_RECORDS_DIR
from datasets.tf_record_utils import _parse_examples


def _tf_record_dataset(dataset_name, ordered=False):
    path = f'{MSASL_TF_RECORDS_DIR}/{dataset_name}'
    files = [f'{path}/{file}' for file in os.listdir(path)]
    num_parallel_reads = 1 if ordered else tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=num_parallel_reads)
    if not ordered:
        options = tf.data.Options()
        options.experimental_deterministic = False
        dataset = dataset.with_options(options)
    return dataset


def _display_images(images):
    for image in images:
        cv2.imshow('Frame', image)
        cv2.waitKey(0)


def inspect_dataset():
    dataset = _tf_record_dataset('train')
    dataset = dataset.batch(2)
    dataset = dataset.map(_parse_examples)
    for frames, labels, signers in dataset.take(1):
        frames = frames.numpy()
        _display_images(frames[0])


if __name__ == '__main__':
    inspect_dataset()
