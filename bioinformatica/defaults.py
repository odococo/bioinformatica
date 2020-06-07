from typing import Union

_defaults = {
    'dataset': "fantom",
    'window_size': 200,
    'dataset_path': "datasets",
    'assembly': "hg19",
    'nucleotides': "actg",
    'batch_size': 1024,
    'epochs': 1000,
    'splits': 50,
    'cell_line': '',
    'region': '',
    'test_size': 0.2,
    'validation_split': 0.0,
    'alpha': 0.1,
    'verbose': False,
    'optimizer': 'nadam',
    'loss': 'binary_crossentropy',
    'boruta_iterations': 10,
    'results_path': '',
    'shuffle': True
}


def set_default(**values):
    _defaults.update(values)


def get_default(key: str, default=None) -> Union[int, str]:
    return _defaults.get(key, default)
