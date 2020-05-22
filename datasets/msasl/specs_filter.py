import json

from datasets.constants import DATASET_NAMES
from datasets.msasl.constants import _MSASL_SPECS_DIR, _MSASL_FILTERED_SPECS_DIR
from datasets.msasl.video_download import _downloaded_video_ids, _extract_video_id


def create_filtered_specs():
    """Filters those rows from the original `MS-ASL` dataset specification files, for which the corresponding videos
    could not be downloaded.

    Stores filtered train, validation and test dataset specification files into the `_MSASL_FILTERED_SPECS_DIR`
    directory.
    """
    videos = _downloaded_video_ids()
    for dataset_name in DATASET_NAMES:
        with open(f'{_MSASL_SPECS_DIR}/MSASL_{dataset_name}.json', 'r') as input:
            dataset = json.load(input)
            filtered_dataset = [it for it in dataset if _extract_video_id(it['url']) in videos]
            with open(f'{_MSASL_FILTERED_SPECS_DIR}/{dataset_name}.json', 'w', encoding='utf-8') as output:
                output.write('[' + ',\n'.join(json.dumps(it) for it in filtered_dataset) + ']')


if __name__ == '__main__':
    create_filtered_specs()
