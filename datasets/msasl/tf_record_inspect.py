import cv2

from datasets.tf_record_utils import tf_record_dataset, _parse_examples


def _display_images(images):
    for image in images:
        cv2.imshow('Frame', image)
        cv2.waitKey(0)


def inspect_dataset():
    dataset = tf_record_dataset('train')
    dataset = dataset.batch(2)
    dataset = dataset.map(_parse_examples)
    for frames, labels, signers in dataset.take(1):
        frames = frames.numpy()
        _display_images(frames[0])


if __name__ == '__main__':
    inspect_dataset()
