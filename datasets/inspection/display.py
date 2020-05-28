import cv2

_FONT = cv2.QT_FONT_NORMAL, 0.5, (60, 60, 60), 1, cv2.LINE_AA


def _display_frames(frames, show_frame_number=True, delay=0):
    """Displays a sequence of frames.

    Arguments:
        frames: The frames to display.
        show_frame_number: Whether to display the frame number of the current frame displayed (defaults to `True`).
        delay: The number of milliseconds an individual frame is displayed (defaults to 0; waiting for any keypress).
    """
    count = len(frames)
    for i, frame in enumerate(frames):
        if show_frame_number:
            frame = cv2.putText(frame, f'{i + 1}:{count}', (5, 20), *_FONT)
        cv2.imshow('Display', frame)
        cv2.waitKey(delay)
    cv2.destroyAllWindows()


def _play_frames(frames, show_frame_number=True, delay=150):
    """Displays a sequence of frames in succession.

    Convenience function with a default play rate for method `_display_frames`.

    Arguments:
        frames: The frames to display in succession.
        show_frame_number: Whether to display the frame number of the current frame displayed (defaults to `True`).
        delay: The number of milliseconds an individual frame is displayed (defaults to 150).
    """
    _display_frames(frames, show_frame_number, delay)
