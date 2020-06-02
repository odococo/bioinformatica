from typing import Union

_defaults = {
    'dataset': "fantom",
    'window_size': 200,
    'dataset_path': "datasets",
    'assembly': "hg19",
    'nucleotides': "actg",
    'batch_size': 1024,
    'splits': 50,
    'cell_line': '',
    'region': ''
}


def set_default(**values):
    _defaults.update(values)


def get_default(key: str, default=None) -> Union[int, str]:
    return _defaults.get(key, default)
