import json
import logging
from pathlib import Path

import cv2
import tensorflow as tf

from datasets.constants import DatasetType, _TF_RECORD_SHARD_SIZE
from datasets.constants import _FRAME_SIZE
from datasets.msasl.constants import _MSASL_FILTERED_SPECS_DIR, _MSASL_VIDEOS_DIR, _MSASL_TF_RECORDS_DIR
from datasets.msasl.video_download import _extract_video_id
from datasets.tf_record_utils import _bytes_feature, _int64_feature
from datasets.utils import _crop_image_to_square, _frame_positions


def _center_ratios(box):
    """Calculates the center of the bounding box.

    The bounding box has the format [y0, x0, y1, x1], where (x0, y0) is the upper-left corner and (x1, y1) is the
    bottom-right corner. The coordinates are normalized into the range between 0 and 1.

    Arguments:
        box: The bounding box surrounding the signer in an image.

    Returns:
        The relative abscissa and ordinate of the center of the bounding box.
    """
    x = (box[1] + box[3]) / 2
    y = (box[0] + box[2]) / 2
    return x, y


def _read_frames(example):
    """Reads the individual frames from the corresponding YouTube video.

    A frame is encoded as a compressed JPEG image instead of an ndarray, as the latter consumes up to 10 times as much
    storage space.

    Arguments:
        example: An example from a dataset specification file.

    Returns:
        The list of frames of the dataset example.
    """
    frames = []
    path = f"{_MSASL_VIDEOS_DIR}/{_extract_video_id(example['url'])}.mp4"
    cap = cv2.VideoCapture(path)
    positions = _frame_positions(example['start'], example['end'])
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos - 1)
        _, frame = cap.read()
        center_x_ratio, center_y_ratio = _center_ratios(example['box'])
        cropped_frame = _crop_image_to_square(frame, center_x_ratio, center_y_ratio)
        _, buffer = cv2.imencode('.jpg', cv2.resize(cropped_frame, _FRAME_SIZE))
        frames.append(buffer.tobytes())
    cap.release()
    return frames


def _serialize_example(example):
    """Serializes a dataset example in the `TFRecord` format.

    Arguments:
        example: An example from a dataset specification file.

    Returns:
        The binary string representation of the `TFRecord`.
    """
    frames = _read_frames(example)
    feature = {
        'frames': _bytes_feature(frames),
        'label': _int64_feature([example['label']]),
        'signer': _int64_feature([example['signer_id']])
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example.SerializeToString()


def write_tf_records():
    """Creates the `TFRecord` files of the `MS-ASL` train, validation and test datasets.

    The `TFRecord` files of the train, validation and test datasets are saved into corresponding subdirectories inside
    the `MSASL_TF_RECORDS_DIR` directory. The examples of each of the train, validation and test datasets are sharded
    into multiple `TFRecord` files. The number of examples a single `TFRecord` file contains is set by the
    `_TF_RECORD_SHARD_SIZE` constant.
    """
    for dataset_type in DatasetType:
        with open(f'{_MSASL_FILTERED_SPECS_DIR}/{dataset_type.value}.json', 'r') as file:
            dataset = json.load(file)
        writer = None
        running_file_number = 0
        running_record_number = 0
        Path(f'{_MSASL_TF_RECORDS_DIR}/{dataset_type.value}').mkdir(exist_ok=True)
        for i, example in enumerate(dataset):
            if i % _TF_RECORD_SHARD_SIZE == 0:
                if writer:
                    writer.close()
                    logging.info(f'{running_record_number} records have been written to {file_name}.')
                    running_record_number = 0
                running_file_number += 1
                file_name = f'{_MSASL_TF_RECORDS_DIR}/{dataset_type.value}/msasl_{dataset_type.value}_{running_file_number:02d}.tfrecord'
                writer = tf.io.TFRecordWriter(file_name)
            serialized_example = _serialize_example(example)
            writer.write(serialized_example)
            running_record_number += 1
        if writer:
            writer.close()
            logging.info(f'{running_record_number} records have been written to {file_name}.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    write_tf_records()
