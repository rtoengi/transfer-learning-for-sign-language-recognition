import os

import cv2
import numpy as np
import tensorflow as tf

from datasets.constants import DatasetName, DatasetType
from datasets.constants import _N_TIME_STEPS
from datasets.msasl.constants import N_CLASSES as MSASL_N_CLASSES
from datasets.signum.constants import N_CLASSES as SIGNUM_N_CLASSES
from datasets.utils import _tf_records_dir


def tf_record_dataset(dataset_name: DatasetName, dataset_type: DatasetType, ordered=False):
    """Returns a `TFRecordDataset` of the requested dataset.

    Arguments:
        dataset_name: The name of the dataset.
        dataset_type: The type of the dataset.
        ordered: Whether the examples should be fetched in order.

    Returns:
        A `TFRecordDataset` of the requested dataset.
    """
    path = f'{_tf_records_dir(dataset_name)}/{dataset_type.value}'
    files = [f'{path}/{file}' for file in os.listdir(path)]
    num_parallel_reads = 1 if ordered else tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=num_parallel_reads)
    if not ordered:
        options = tf.data.Options()
        options.experimental_deterministic = False
        dataset = dataset.with_options(options)
    return dataset


def _dataset_counts(dataset_name: DatasetName):
    """Returns the sizes of the `dataset_name` train, validation and test datasets.

    Arguments:
        dataset_name: The name of the dataset.

    Returns:
        A dictionary with an entry of the size for each of the train, validation and test datasets.
    """
    counts = {}
    for dataset_type in DatasetType:
        dataset = tf_record_dataset(dataset_name, dataset_type)
        dataset = dataset.batch(64)
        dataset = dataset.prefetch(1)
        counts[dataset_type.value] = 0
        for records in dataset:
            counts[dataset_type.value] += len(records)
    return counts


def _bytes_feature(bytes_list):
    """Converts a list of bytestrings into a protocol buffer feature message.

    Returns:
        A protocol buffer feature message.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=bytes_list))


def _float_feature(float_list):
    """Converts a list of floats into a protocol buffer feature message.

    Returns:
        A protocol buffer feature message.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=float_list))


def _int64_feature(int64_list):
    """Converts a list of ints into a protocol buffer feature message.

    Returns:
        A protocol buffer feature message.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))


def _decode_jpeg(bytestring):
    """Decodes a compressed JPEG image into an ndarray.

    Arguments:
        bytestring: The binary string representation of the compressed JPEG image.

    Returns:
        The ndarray representation of the image using the uint8 data type for the channel values.
    """
    image = np.frombuffer(bytestring, np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)


def _transform_frames_for_inspection(examples):
    """Transforms a batch of example frames to be consumed for inspection.

    An individual frame is a 3D tensor consisting of RGB uint8 values within the range [0, 255] represented in the
    `channels_last` data format ([height, width, channels]).

    Arguments:
        examples: A 5D tensor representing a batch of example frames in the [batch, frames, height, width, channels]
        format.

    Returns:
        The transformed batch of example frames.
    """
    return np.array([[_decode_jpeg(frame) for frame in example] for example in examples.numpy()])


def _transform_frames_for_model(examples):
    """Transforms a batch of example frames to be consumed by a model.

    An individual frame is a 3D tensor consisting of RGB float32 values within the range [-1.0, 1.0] represented in the
    `channels_last` data format ([height, width, channels]).

    Arguments:
        examples: A 5D tensor representing a batch of example frames in the [batch, frames, height, width, channels]
        format.

    Returns:
        The transformed batch of example frames.
    """
    frames = np.array([[_decode_jpeg(frame) for frame in example] for example in examples.numpy()])
    frames = frames.astype(np.float32, copy=False)
    frames /= 127.5
    frames -= 1.0
    return frames


_FEATURES = {
    'frames': tf.io.FixedLenFeature([_N_TIME_STEPS], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'signer': tf.io.FixedLenFeature([], tf.int64)
}


def transform_for_inspection(examples):
    """Transforms a batch of examples to be consumed for inspection.

    The returned frames are represented as RGB uint8 values within the range [0, 255], and the labels and signers are
    represented as their corresponding indices.

    Arguments:
        examples: A batch of serialized `TFRecord` examples.

    Returns:
        A tuple of batches of frames, labels and signers.
    """
    parsed_examples = tf.io.parse_example(examples, _FEATURES)

    frames = tf.py_function(_transform_frames_for_inspection, [parsed_examples['frames']], tf.uint8)
    labels = parsed_examples['label']
    signers = parsed_examples['signer']

    return frames, labels, signers


def transform_for_msasl_model(examples):
    """Transforms a batch of `MS-ASL` dataset examples to be consumed by a model.

    The returned frames are represented as RGB float32 values within the range [-1.0, 1.0], and the labels are one-hot
    encoded with a depth of `datasets.msasl.constants.N_CLASSES`.

    Arguments:
        examples: A batch of serialized `TFRecord` examples.

    Returns:
        A tuple of batches of frames and labels.
    """
    parsed_examples = tf.io.parse_example(examples, _FEATURES)

    frames = tf.py_function(_transform_frames_for_model, [parsed_examples['frames']], tf.float32)
    labels = tf.one_hot(parsed_examples['label'], MSASL_N_CLASSES)

    return frames, labels


def transform_for_signum_model(examples):
    """Transforms a batch of `SIGNUM` dataset examples to be consumed by a model.

    The returned frames are represented as RGB float32 values within the range [-1.0, 1.0], and the labels are one-hot
    encoded with a depth of `datasets.signum.constants.N_CLASSES`.

    Arguments:
        examples: A batch of serialized `TFRecord` examples.

    Returns:
        A tuple of batches of frames and labels.
    """
    parsed_examples = tf.io.parse_example(examples, _FEATURES)

    frames = tf.py_function(_transform_frames_for_model, [parsed_examples['frames']], tf.float32)
    labels = tf.one_hot(parsed_examples['label'], SIGNUM_N_CLASSES)

    return frames, labels
