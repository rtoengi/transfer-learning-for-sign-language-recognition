import tensorflow as tf


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
