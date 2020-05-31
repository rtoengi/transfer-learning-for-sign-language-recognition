import numpy as np

from datasets.constants import DatasetName, DatasetType, _N_TIME_STEPS
from datasets.msasl.constants import _MSASL_TF_RECORDS_DIR
from datasets.signum.constants import _SIGNUM_TF_RECORDS_DIR


def _frame_positions(start: int, end: int):
    """Samples frame numbers in the range between `start` and `end`.

    `_N_TIME_STEPS` number of samples are drawn uniformly in the range between `start` and `end`. If `_N_TIME_STEPS` is
    greater than the range that is drawn from, then sampling is done with replacement. The samples are sorted in
    ascending order.

    Arguments:
        start: The start frame number, inclusive.
        end: The end frame number, inclusive.

    Returns:
        The sorted array of frame numbers between `start` and `end`.
    """
    range_ = end - start + 1
    positions = np.random.choice(range_, _N_TIME_STEPS, replace=range_ < _N_TIME_STEPS) + start
    positions.sort()
    return positions


def _crop_image_to_square(image, center_x_ratio=0.5, center_y_ratio=0.5):
    """Crops an image along its longer side to get a square image.

    The image is cropped along its longer side so that the resulting image has the same width and height. If the image
    is wider than high, then the horizontal cropping is centered around `center_x_ratio`. On the other hand, if the
    image is higher than wide, then the vertical cropping is centered around `center_y_ratio`. Should the cropping
    around the center ratios exceed the boundaries of the image, then the cropping sticks to the corresponding boundary.
    A square image is left unchanged.

    Arguments:
        image: The ndarray image to be cropped.
        center_x_ratio: The relative abscissa the horizontal cropping is centered around (float between 0 and 1).
        center_y_ratio: The relative ordinate the vertical cropping is centered around (float between 0 and 1).

    Returns:
        A square ndarray image.
    """
    height = image.shape[0]
    width = image.shape[1]
    if width > height:
        mid_x = int(width * center_x_ratio)
        half_height = int(height / 2)
        if center_x_ratio <= 0.5:
            x = max(0, mid_x - half_height)
            return image[:, x:x + height]
        else:
            x = min(width, mid_x + half_height)
            return image[:, x - height:x]
    elif height > width:
        mid_y = int(height * center_y_ratio)
        half_width = int(width / 2)
        if center_y_ratio <= 0.5:
            y = max(0, mid_y - half_width)
            return image[y:y + width]
        else:
            y = min(height, mid_y + half_width)
            return image[y - width:y]
    else:
        return image


_TF_RECORDS_DIR_DICT = {
    DatasetName.MSASL: _MSASL_TF_RECORDS_DIR,
    DatasetName.SIGNUM: _SIGNUM_TF_RECORDS_DIR
}


def _tf_records_dir(dataset_name: DatasetName):
    """Returns the path to the directory storing the `TFRecord` files of the requested dataset.

    Arguments:
        dataset_name: The name of the dataset.

    Returns:
        The absolute path to the directory storing the `TFRecord` files of the requested dataset.
    """
    return _TF_RECORDS_DIR_DICT[dataset_name]


def _display_dataset_counts(dataset_name: DatasetName, counts):
    """Displays the sizes of the `dataset_name` train, validation and test datasets."""
    line_width = 27
    print(f'{dataset_name.name} dataset counts')
    print('=' * line_width)
    for dataset_type in DatasetType:
        count = counts[dataset_type.value]
        print(f"{dataset_type.name + ':':13s} {count:5d} ({count / sum(counts.values()) * 100:02.1f}%)")
    print('=' * line_width)
    print(f"{'Total:':13s} {sum(counts.values()):5d} (100%)")
