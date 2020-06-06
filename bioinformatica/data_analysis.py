from multiprocessing import cpu_count
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from boruta import BorutaPy
from prince import MFA
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE as STSNE
from tqdm import tqdm

from .defaults import get_default


def _pca(data: Union[pd.DataFrame, np.ndarray], n_components: int = 2) -> np.ndarray:
    return PCA(n_components=n_components, random_state=42).fit_transform(data)


def _mfa(data: pd.DataFrame, n_components: int = 2, nucleotides: str = get_default('nucleotides')) -> np.ndarray:
    return MFA(groups={
        nucleotide: [
            column
            for column in data.columns
            if nucleotide in column
        ]
        for nucleotide in nucleotides
    }, n_components=n_components, random_state=42).fit_transform(data)


def get_filtered_with_boruta(epigenomes: pd.DataFrame, labels: pd.DataFrame,
                             cell_line: str, region: str) -> pd.DataFrame:
    def get_features_filter(data: pd.DataFrame, label: pd.DataFrame, name: str) -> BorutaPy:
        boruta_selector = BorutaPy(
            RandomForestClassifier(n_jobs=cpu_count(), class_weight='balanced', max_depth=5),
            n_estimators='auto',
            verbose=2,
            alpha=0.05,  # p_value
            max_iter=get_default('boruta_iterations'),  # In practice one would run at least 100-200 times
            random_state=42
        )
        boruta_selector.fit(data.values, label.values.ravel())
        return boruta_selector

    return pd.DataFrame(get_features_filter(
        data=epigenomes,
        label=labels,
        name=f"{cell_line}/{region}"
    ).transform(epigenomes.values))


def get_tasks(epigenomes: Dict[str, pd.DataFrame], labels: Dict[str, pd.DataFrame], sequences: Dict[str, pd.DataFrame]):
    tasks = {
        "x": [
            *[
                val.values
                for val in epigenomes.values()
            ],
            *[
                val.values
                for val in sequences.values()
            ]
        ],
        "y": [
            *[
                val.values.ravel()
                for val in labels.values()
            ],
            *[
                val.values.ravel()
                for val in labels.values()
            ]
        ],
        "titles": [
            "Epigenomes promoters",
            "Epigenomes enhancers",
            "Sequences promoters",
            "Sequences enhancers"
        ]
    }

    return tasks['x'], tasks['y'], tasks['title']


def _sklearn_tsne(data: np.ndarray, perplexity: int, dimensionality_threshold: int = 50):
    if data.shape[1] > dimensionality_threshold:
        data = _pca(data, n_components=dimensionality_threshold)
    return STSNE(perplexity=perplexity, n_jobs=cpu_count(), random_state=42).fit_transform(data)


def show_decomposed_data(xs, ys, titles):
    colors = np.array([
        "tab:blue",
        "tab:orange",
    ])

    def show_pca():
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(32, 16))
        for x, y, title, axis in tqdm(zip(xs, ys, titles, axes.flatten()), desc="Computing PCAs", total=len(xs)):
            axis.scatter(*_pca(x).T, s=1, color=colors[y])
            axis.xaxis.set_visible(False)
            axis.yaxis.set_visible(False)
            axis.set_title(f"PCA decomposition - {title}")
        fig.tight_layout()
        plt.show()

    def show_tsne():
        for perpexity in tqdm((30, 40), desc="Running perplexities"):
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(40, 20))
            for x, y, title, axis in tqdm(zip(xs, ys, titles, axes.flatten()), desc="Computing TSNEs", total=len(xs)):
                axis.scatter(*_sklearn_tsne(x, perplexity=perpexity).T, s=1, color=colors[y])
                axis.xaxis.set_visible(False)
                axis.yaxis.set_visible(False)
                axis.set_title(f"TSNE decomposition - {title}")
            fig.tight_layout()
            plt.show()

    show_pca()
    show_tsne()
