from typing import Tuple

import numpy as np
import pandas as pd
from epigenomic_dataset import load_epigenomes
from keras_bed_sequence import BedSequence
from mixed_sequence import MixedSequence
#from keras_mixed_sequence import MixedSequence
from tensorflow.keras.utils import Sequence
from ucsc_genomes_downloader import Genome

from defaults import get_default


def download_data(cell_line: str, region: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download data from Fantom

    :param cell_line
    :param region

    :return tuple dividing promoters and enhancer data and labels
    """
    return load_epigenomes(
        cell_line=cell_line,
        dataset=get_default('dataset'),
        regions=region,
        window_size=get_default('window_size'),
        root=get_default('dataset_path')
    )


def to_bed(data: pd.DataFrame) -> pd.DataFrame:
    """Return bed coordinates from given dataset.

    :param data

    :return dataframe with bed coordinates
    """
    return data.reset_index()[data.index.names]


def get_genome() -> Genome:
    """Download genome or retrieve it if given path"""
    return Genome(get_default('assembly'))


def one_hot_encode(data: pd.DataFrame, genome: Genome) -> np.ndarray:
    """Set to one only the nucletoide, zero the others

    :param data
    :param genome
    """
    return np.array(BedSequence(
        genome,
        bed=to_bed(data),
        nucleotides=get_default('nucleotides'),
        batch_size=1
    )).reshape(-1, get_default('window_size') * len(get_default('nucleotides'))).astype(int)


def to_dataframe(data: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        data,
        columns=[
            f"{i}{nucleotide}"
            for i in range(get_default('window_size'))
            for nucleotide in get_default('nucleotides')
        ]
    )


def get_sequences(epigenomes: pd.DataFrame) -> pd.DataFrame:
    return to_dataframe(one_hot_encode(epigenomes, get_genome()))


def get_holdout(train: np.ndarray, test: np.ndarray,
                bed: pd.DataFrame, labels: pd.DataFrame) -> Tuple[Sequence, Sequence]:
    genome = get_genome()
    batch_size = get_default('batch_size')
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


def data_retrieval(cell_line: str, region: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print('Downloading data...')
    epigenomes, labels = download_data(cell_line, region)
    print('Finished!')
    return epigenomes, labels
