import os

import cv2
import numpy as np
import tensorflow as tf

# from tensorflow.keras.applications.inception_v3 import preprocess_input
from datasets.constants import DatasetType
from datasets.constants import _N_TIME_STEPS
from datasets.msasl.constants import MSASL_TF_RECORDS_DIR


def tf_record_dataset(dataset_type: DatasetType, ordered=False):
    path = f'{MSASL_TF_RECORDS_DIR}/{dataset_type.value}'
    files = [f'{path}/{file}' for file in os.listdir(path)]
    num_parallel_reads = 1 if ordered else tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=num_parallel_reads)
    if not ordered:
        options = tf.data.Options()
        options.experimental_deterministic = False
        dataset = dataset.with_options(options)
    return dataset


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
    img = np.frombuffer(bytestring, np.uint8)
    return cv2.imdecode(img, cv2.IMREAD_COLOR)


def _decode_frames(examples):
    return np.array([[_decode_jpeg(frame) for frame in example] for example in examples.numpy()])


_features = {
    'frames': tf.io.FixedLenFeature([_N_TIME_STEPS], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'signer': tf.io.FixedLenFeature([], tf.int64)
}


def _parse_examples(examples):
    parsed_examples = tf.io.parse_example(examples, _features)

    frames = tf.py_function(_decode_frames, [parsed_examples['frames']], tf.uint8)
    labels = parsed_examples['label']
    signers = parsed_examples['signer']

    return frames, labels, signers
