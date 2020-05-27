import json
import os
from logging import error

import youtube_dl as ydl
from youtube_dl.utils import DownloadError

from datasets.constants import DatasetType
from datasets.msasl.constants import _MSASL_SPECS_DIR, _MSASL_VIDEOS_DIR


def _video_urls():
    """Returns the URLs of videos that have not yet been downloaded.

    All the URLs in the train, validation and test dataset files of the `MS-ASL` dataset are taken into account.

    Returns:
        The set of URLs of videos that have not yet been downloaded.
    """
    urls = set()
    for dataset_type in DatasetType:
        with open(f'{_MSASL_SPECS_DIR}/MSASL_{dataset_type.value}.json', 'r') as file:
            dataset = json.load(file)
        urls = urls.union({it['url'] for it in dataset})
    return {url for url in urls if _extract_video_id(url) not in _downloaded_video_ids()}


def _extract_video_id(url=''):
    """Extracts the video ID from a YouTube URL.

    Arguments:
        url: A string representing a YouTube URL (e.g. 'https://www.youtube.com/watch?v=S2cqitZ0qes').

    Returns:
        The video ID of a YouTube URL (e.g. 'S2cqitZ0qes').
    """
    index = url.index('?v=') + 3
    return url[index:]


def _downloaded_video_ids():
    """Returns the IDs of videos that have already been downloaded.

    Can be utilized to prevent the re-downloading of videos in case the download process is executed multiple times.

    Returns:
        The set of IDs of videos that have already been downloaded.
    """
    videos = os.listdir(_MSASL_VIDEOS_DIR)
    return set([video[:-4] for video in videos])


def _download_video(url):
    """Downloads a YouTube video at `url`.

    The video is stored into the `_MSASL_VIDEOS_DIR` directory.

    Arguments:
        url: A string representing a YouTube URL.
    """
    ydl_opts = {'outtmpl': _MSASL_VIDEOS_DIR + '/%(id)s.%(ext)s'}
    with ydl.YoutubeDL(ydl_opts) as video:
        try:
            video.download([url])
        except DownloadError:
            error(f'Video at {url} could not be downloaded.')


def download_videos():
    """Downloads the `MS-ASL` YouTube videos.

    The `MS-ASL` dataset consists of URLs of YouTube videos. Such a video may contain multiple sign gestures, which
    means that different dataset examples may refer to the same video. This function downloads all the videos of the
    train, validation and test dataset specification files into the `_MSASL_VIDEOS_DIR` directory.
    """
    for url in _video_urls():
        _download_video(url)


if __name__ == '__main__':
    download_videos()
