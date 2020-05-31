import pandas as pd
import numpy as np

from typing import Dict, Tuple

from epigenomic_dataset import load_epigenomes
from ucsc_genomes_downloader import Genome
from keras_bed_sequence import BedSequence
from tensorflow.keras.utils import Sequence
from keras_mixed_sequence import MixedSequence

_defaults = {
    'dataset': "fantom",
    'window_size': 200,
    'dataset_path': "datasets",
    'assembly': "hg19",
    'nucleotides': "actg",
    'batch_size': 1024
}


def set_default(**values):
    _defaults.update(values)


def download_data(cell_line: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Download data from Fantom

    :param cell_line

    :return tuple dividing promoters and enhancer data and labels
    """
    regions = ["promoters", "enhancers"]
    epigenomes = {}
    labels = {}
    for region in regions:
        epigenome, label = load_epigenomes(
            cell_line=cell_line,
            dataset=_defaults['dataset'],
            regions=region,
            window_size=_defaults['window_size'],
            root=_defaults['dataset_path']
        )
        epigenomes.update({region: epigenome})
        labels.update({region: label})
    return epigenomes, labels


def to_bed(data: pd.DataFrame) -> pd.DataFrame:
    """Return bed coordinates from given dataset.

    :param data

    :return dataframe with bed coordinates
    """
    return data.reset_index()[data.index.names]


def get_genome() -> Genome:
    """Download genome or retrieve it if given path"""
    return Genome(_defaults['assembly'])


def one_hot_encode(data: pd.DataFrame, genome: Genome) -> np.ndarray:
    """Set to one only the nucletoide, zero the others

    :param data
    :param genome
    """
    return np.array(BedSequence(
        genome,
        bed=to_bed(data),
        nucleotides=_defaults['nucleotides'],
        batch_size=1
    )).reshape(-1, _defaults['window_size'] * len(_defaults['nucleotides'])).astype(int)


def to_dataframe(data: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        data,
        columns=[
            f"{i}{nucleotide}"
            for i in range(_defaults['window_size'])
            for nucleotide in _defaults['nucleotides']
        ]
    )


def get_sequences(epigenomes: Dict[str, pd.DataFrame], genome: Genome) -> Dict[str, pd.DataFrame]:
    return {
        region: to_dataframe(one_hot_encode(data, genome))
        for region, data in epigenomes.items()
    }


def get_holdout(train: np.ndarray, test: np.ndarray,
                bed: pd.DataFrame, labels: np.ndarray) -> Tuple[Sequence, Sequence]:
    genome = get_genome()
    batch_size = _defaults['batch_size']
    return (
        MixedSequence(
            x=BedSequence(genome, bed.iloc[train], batch_size=batch_size),
            y=labels[train],
            batch_size=batch_size
        ),
        MixedSequence(
            x=BedSequence(genome, bed.iloc[test], batch_size=batch_size),
            y=labels[test],
            batch_size=batch_size
        )
    )


def data_retrieval(cell_line: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    print('Downloading data...')
    epigenomes, labels = download_data(cell_line)
    print('Downloading genome...')
    genome = get_genome()
    print('Getting dataframe...')
    sequences = get_sequences(epigenomes, genome)
    print('Finished!')
    return epigenomes, labels, sequences
