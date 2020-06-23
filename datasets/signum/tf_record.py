import logging
import os
from operator import itemgetter
from pathlib import Path

import cv2
import tensorflow as tf

from datasets.constants import DatasetType, _FRAME_SIZE
from datasets.signum.constants import _SIGNUM_IMAGES_DIR, _SIGNUM_TF_RECORDS_DIR, _N_FRAMES_PER_EXAMPLE
from datasets.tf_record_utils import _bytes_feature, _int64_feature
from datasets.utils import _crop_image_to_square
from datasets.utils import _frame_positions


def _dataset_type_dir(signer):
    """Returns the directory name of the corresponding dataset type.

    There is a `TFRecord` file written for each of the 25 signers. The `TFRecord` files of the first 17 signers are
    assigned to the train dataset, the `TFRecord` files of the next 4 signers are assigned to the validation dataset,
    and the `TFRecord` files of the last 4 signers are assigned to the test dataset.

    Arguments:
        signer: The index of the signer.

    Returns:
        The directory name of the corresponding dataset type.
    """
    if signer > 20:
        return DatasetType.TEST.value
    elif signer > 16:
        return DatasetType.VALIDATION.value
    else:
        return DatasetType.TRAIN.value


def _read_frames(example_path):
    """Reads the individual frames out of the list of images of the dataset example.

    A frame is encoded as a compressed JPEG image instead of an ndarray, as the latter consumes up to 10 times as much
    storage space.

    Arguments:
        example_path: The path to the dataset example.

    Returns:
        The list of frames of the dataset example.
    """
    positions = _frame_positions(0, _N_FRAMES_PER_EXAMPLE - 1)
    file_names = itemgetter(*positions)(os.listdir(example_path))
    frames = []
    for file_name in file_names:
        frame = cv2.imread(f'{example_path}/{file_name}')
        cropped_frame = _crop_image_to_square(frame)
        _, buffer = cv2.imencode('.jpg', cv2.resize(cropped_frame, _FRAME_SIZE))
        frames.append(buffer.tobytes())
    return frames


def _serialize_example(example_path, label, signer):
    """Serializes a dataset example in the `TFRecord` format.

    Arguments:
        example_path: The path to the dataset example.
        label: The index of the label.
        signer: The index of the signer.

    Returns:
        The binary string representation of the `TFRecord`.
    """
    frames = _read_frames(example_path)
    feature = {
        'frames': _bytes_feature(frames),
        'label': _int64_feature([label]),
        'signer': _int64_feature([signer])
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example.SerializeToString()


def write_tf_records():
    """Creates the `TFRecord` files of the `SIGNUM` train, validation and test datasets.

    The `TFRecord` files of the train, validation and test datasets are saved into corresponding subdirectories inside
    the `_SIGNUM_TF_RECORDS_DIR` directory. There is one `TFRecord` file for each signer.
    """
    signer_dirs = os.listdir(_SIGNUM_IMAGES_DIR)
    for signer, signer_dir in enumerate(signer_dirs):
        path = f'{_SIGNUM_TF_RECORDS_DIR}/{_dataset_type_dir(signer)}'
        Path(path).mkdir(exist_ok=True)
        file_name = f'{path}/signum_{signer + 1:02d}.tfrecord'
        with tf.io.TFRecordWriter(file_name) as writer:
            running_record_number = 0
            example_dirs = os.listdir(f'{_SIGNUM_IMAGES_DIR}/{signer_dir}')
            for label, example_dir in enumerate(example_dirs):
                example_path = f'{_SIGNUM_IMAGES_DIR}/{signer_dir}/{example_dir}'
                serialized_example = _serialize_example(example_path, label, signer)
                writer.write(serialized_example)
                running_record_number += 1
            logging.info(f'{running_record_number} records have been written to {file_name}.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    write_tf_records()
