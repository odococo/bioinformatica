from multiprocessing import cpu_count

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from boruta import BorutaPy
from prince import MFA
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from sklearn.manifold import TSNE as STSNE


def get_filtered_with_boruta(epigenomes:  pd.DataFrame,
                             labels: Dict[str, pd.DataFrame], cell_line: str, region:str) -> pd.DataFrame:
    def get_features_filter(data: pd.DataFrame, label: pd.DataFrame, name: str) -> BorutaPy:
        boruta_selector = BorutaPy(
            RandomForestClassifier(n_jobs=cpu_count(), class_weight='balanced', max_depth=5),
            n_estimators='auto',
            verbose=2,
            alpha=0.05,  # p_value
            max_iter=10,  # In practice one would run at least 100-200 times
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
    def check_tasks(xs, ys, titles: List[str]) -> Tuple:
        assert len(xs) == len(ys) == len(titles)
        for x, y in zip(xs, ys):
            assert x.shape[0] == y.shape[0]
        return xs, ys, titles

    tasks = {
        "x": [
            *[
                val.values
                for val in epigenomes.values()
            ],
            *[
                val.values
                for val in sequences.values()
            ],
            pd.concat(sequences.values()).values,
            pd.concat(sequences.values()).values,
            *[
                np.hstack([
                    pca(epigenomes[region], n_components=25),
                    mfa(sequences[region], n_components=25)
                ])
                for region in epigenomes
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
            ],
            pd.concat(labels.values()).values.ravel(),
            np.vstack([np.ones_like(labels["promoters"]), np.zeros_like(labels["enhancers"])]).ravel(),
            *[
                val.values.ravel()
                for val in labels.values()
            ],
        ],
        "titles": [
            "Epigenomes promoters",
            "Epigenomes enhancers",
            "Sequences promoters",
            "Sequences enhancers",
            "Sequences active regions",
            "Sequences regions types",
            "Combined promoters data",
            "Combined enhancers data"
        ]
    }

    return check_tasks(tasks['x'], tasks['y'], tasks['title'])


def get_decomposed_data(xs, ys, titles):
    def pca(data: np.ndarray, n_components: int = 2) -> np.ndarray:
        return PCA(n_components=n_components, random_state=42).fit_transform(data)

    def mfa(data: pd.DataFrame, n_components: int = 2, nucleotides: str = 'actg') -> np.ndarray:
        return MFA(groups={
            nucleotide: [
                column
                for column in data.columns
                if nucleotide in column
            ]
            for nucleotide in nucleotides
        }, n_components=n_components, random_state=42).fit_transform(data)

    def sklearn_tsne(x: np.ndarray, perplexity: int, dimensionality_threshold: int = 50):
        if x.shape[1] > dimensionality_threshold:
            x = pca(x, n_components=dimensionality_threshold)
        return STSNE(perplexity=perplexity, n_jobs=cpu_count(), random_state=42).fit_transform(x)

    #def ulyanov_tsne(x: np.ndarray, perplexity: int, dimensionality_threshold: int = 50, n_components: int = 2):
    #    if x.shape[1] > dimensionality_threshold:
    #        x = pca(x, n_components=dimensionality_threshold)
    #    return UTSNE(n_components=n_components, perplexity=perplexity, n_jobs=cpu_count(), random_state=42,
    #                 verbose=True).fit_transform(x)

    #def cannylab_tsne(x: np.ndarray, perplexity: int, dimensionality_threshold: int = 50):
    #    if x.shape[1] > dimensionality_threshold:
    #        x = pca(x, n_components=dimensionality_threshold)
    #    return CTSNE(perplexity=perplexity, random_seed=42).fit_transform(x)

    #def nystroem(x: np.array) -> np.array:
    #    return Nystroem(random_state=42, n_components=300).fit_transform(x)

    #def monte_carlo(x: np.array) -> np.array:
    #    return RBFSampler(random_state=42, n_components=300).fit_transform(x)

    #def linear(x: np.array) -> np.array:
    #    return x


def show_decomposed_data():
    colors = np.array([
        "tab:blue",
        "tab:orange",
    ])

    def show_pca():
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(32, 16))
        for x, y, title, axis in tqdm(zip(xs, ys, titles, axes.flatten()), desc="Computing PCAs", total=len(xs)):
            axis.scatter(*pca(x).T, s=1, color=colors[y])
            axis.xaxis.set_visible(False)
            axis.yaxis.set_visible(False)
            axis.set_title(f"PCA decomposition - {title}")
        plt.show()

    def show_tsne():
        for perpexity in tqdm((30, 40, 50, 100, 500, 5000), desc="Running perplexities"):
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(40, 20))
            for x, y, title, axis in tqdm(zip(xs, ys, titles, axes.flatten()), desc="Computing TSNEs", total=len(xs)):
                axis.scatter(*sklearn_tsne(x, perplexity=perpexity).T, s=1, color=colors[y])
                axis.xaxis.set_visible(False)
                axis.yaxis.set_visible(False)
                axis.set_title(f"TSNE decomposition - {title}")
            fig.tight_layout()
            plt.show()
