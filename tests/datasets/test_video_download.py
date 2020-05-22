from datasets.msasl.video_download import _extract_video_id


def test__extract_video_id():
    assert _extract_video_id('https://www.youtube.com/watch?v=S2cqitZ0qes') == 'S2cqitZ0qes'
