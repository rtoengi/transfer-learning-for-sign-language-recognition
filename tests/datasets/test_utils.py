import numpy as np

from datasets.utils import _crop_image_to_square


def test__crop_image_to_square():
    # Don't crop square image
    image = np.random.rand(5, 5, 3)
    assert np.array_equal(image, _crop_image_to_square(image))

    image = np.random.rand(5, 10, 3)

    # Crop horizontally on the left side
    assert np.array_equal(image[:, 2:7, :], _crop_image_to_square(image, 0.4, 0.5))

    # Crop horizontally at the left boundary
    assert np.array_equal(image[:, :5, :], _crop_image_to_square(image, 0.1, 0.5))

    # Crop horizontally on the right side
    assert np.array_equal(image[:, 3:8, :], _crop_image_to_square(image, 0.6, 0.5))

    # Crop horizontally at the right boundary
    assert np.array_equal(image[:, -5:, :], _crop_image_to_square(image, 0.9, 0.5))

    image = np.random.rand(10, 5, 3)

    # Crop vertically on the upper side
    assert np.array_equal(image[2:7, :, :], _crop_image_to_square(image, 0.5, 0.4))

    # Crop vertically at the upper boundary
    assert np.array_equal(image[:5, :, :], _crop_image_to_square(image, 0.5, 0.1))

    # Crop vertically on the lower side
    assert np.array_equal(image[3:8, :, :], _crop_image_to_square(image, 0.5, 0.6))

    # Crop vertically at the lower boundary
    assert np.array_equal(image[-5:, :, :], _crop_image_to_square(image, 0.5, 0.9))
