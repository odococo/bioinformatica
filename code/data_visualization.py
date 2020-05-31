import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple

from scipy.stats import pearsonr, spearmanr, entropy
from sklearn.impute import KNNImputer
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm


def overfitting_risk(epigenomes: Dict[str, pd.DataFrame], threshold: int = 1) -> bool:
    valid = False
    for region, data in epigenomes.items():
        rate = data[0] / data[1]
        print(f'Rate for {region} is {rate}')
        valid = valid or (rate < threshold)
    return valid


def nan_check(epigenomes: Dict[str, pd.DataFrame]) -> None:
    for region, x in epigenomes.items():
        print("\n".join((
            f"Nan values report for {region} data:",
            f"In the document there are {x.isna().values.sum()} NaN values out of {x.values.size} values.",
            f"The sample with most values has {x.isna().sum(axis=0).max()} NaN values out of {x.shape[1]} values.",
            f"The feature with most values has {x.isna().sum().max()} NaN values out of {x.shape[0]} values."
        )))
        print("=" * 80)


def fit_constant(data: pd.DataFrame, value: int) -> pd.DataFrame:
    return data.fillna(value)


def fit_media(data: pd.DataFrame) -> pd.DataFrame:
    return fit_constant(data, data.mean())


def fit_median(data: pd.DataFrame) -> pd.DataFrame:
    return fit_constant(data, data.median())


def fit_mode(data: pd.DataFrame) -> pd.DataFrame:
    return fit_constant(data, data.mode())


def fit_neighbours(data: pd.DataFrame, neighbours: int = 5) -> pd.DataFrame:
    return pd.DataFrame(KNNImputer(n_neighbours=neighbours).fit_transform(data.values),
                        columns=data.columns,
                        index=data.index
                        )


def fit_missing(epigenomes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    for region, data in epigenomes.items():
        epigenomes[region] = fit_neighbours(data)
    return epigenomes


def check_class_balance(labels: Dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))

    for axis, (region, y) in zip(axes.ravel(), labels.items()):
        y.hist(ax=axis, bins=3)
        axis.set_title(f"Classes count in {region}")
    fig.show()


def drop_constant_features(epigenomes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    def drop(df: pd.Dataframe) -> pd.DataFrame:
        return df.loc[:, (df != df.iloc[0]).any()]

    for region, data in epigenomes.items():
        result = drop(data)
        if data.shape[1] != result.shape[1]:
            print(f"Features in {region} were constant and had to be dropped!")
            epigenomes[region] = result
        else:
            print(f"No constant features were found in {region}!")
    return epigenomes


def apply_z_scoring(epigenomes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    def robust_zscoring(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            RobustScaler().fit_transform(df.values),
            columns=df.columns,
            index=df.index
        )

    return {region: robust_zscoring(data) for region, data in epigenomes.items()}


def drop_uncorrelated(epigenomes: Dict[str, pd.DataFrame], labels: Dict[str, pd.DataFrame],
                      p_value_threshold: float = 0.01, correlation_threshold: float = 0.05) -> Dict[str, pd.DataFrame]:
    uncorrelated = {region: set() for region in epigenomes}

    def pearson():
        for region, data in epigenomes.items():
            for column in tqdm(data.columns, desc=f"Running Pearson test for {region}", dynamic_ncols=True,
                               leave=False):
                correlation, p_value = pearsonr(data[column].values.ravel(), labels[region].values.ravel())
                if p_value > p_value_threshold:
                    print(region, column, correlation)
                    uncorrelated[region].add(column)

    def spearman():
        for region, data in epigenomes.items():
            for column in tqdm(data.columns, desc=f"Running Spearman test for {region}", dynamic_ncols=True,
                               leave=False):
                correlation, p_value = spearmanr(data[column].values.ravel(), labels[region].values.ravel())
                if p_value > p_value_threshold:
                    print(region, column, correlation)
                uncorrelated[region].add(column)

    def mine():
        for region, data in epigenomes.items():
            for column in tqdm(uncorrelated[region], desc=f"Running MINE test for {region}", dynamic_ncols=True,
                               leave=False):
                mine = MINE()
                mine.compute_score(data[column].values.ravel(), labels[region].values.ravel())
                score = mine.mic()
                if score < correlation_threshold:
                    print(region, column, score)
                else:
                    uncorrelated[region].remove(column)

    def drop():
        for region, data in epigenomes.items():
            epigenomes[region] = data.drop(columns=[col for col in uncorrelated[region] if col in x.columns])
        return epigenomes

    pearson()
    spearman()
    mine()
    return drop()


def drop_too_correlated(epigenomes: Dict[str, pd.DataFrame], p_value_threshold: float = 0.01,
                        correlation_threshold: float = 0.95) -> Dict[str, pd.DataFrame]:
    extremely_correlated = {region: set() for region in epigenomes}
    scores = {region: [] for region in epigenomes}

    def pearson():
        for region, data in epigenomes.items():
            for i, column in tqdm(
                    enumerate(data.columns),
                    total=len(data.columns),
                    desc=f"Running Pearson test for {region}",
                    dynamic_ncols=True,
                    leave=False):
                for feature in data.columns[i + 1:]:
                    correlation, p_value = pearsonr(data[column].values.ravel(), data[feature].values.ravel())
                    correlation = np.abs(correlation)
                    scores[region].append((correlation, column, feature))
                    if p_value < p_value_threshold and correlation > correlation_threshold:
                        print(region, column, feature, correlation)
                        if entropy(data[column]) > entropy(data[feature]):
                            extremely_correlated[region].add(feature)
                        else:
                            extremely_correlated[region].add(column)

    return {region: sorted(score, key=lambda x: np.abs(x[0]), reverse=True) for region, score in scores.items()}


def show(epigenomes: Dict[str, pd.DataFrame], labels: Dict[str, pd.DataFrame], scores: Dict[str, List[Tuple]]):
    def get_data(region: str, how_many: int, from_start: bool = True) -> List[Tuple]:
        data = scores[region][:how_many] if from_start else scores[region][-how_many:]
        data = list(zip(data))[1:]
        columns = list(set([column for row in data for column in row]))
        return columns

    def plot(how_many: int, correlated: bool) -> None:
        for region, data in epigenomes.items():
            print(f"Most {'correlated' if correlated else 'uncorrelated'} features from {region} epigenomes")
            sns.pairplot(pd.concat([
                data[get_data(region, how_many, correlated)],
                labels[region],
            ], axis=1), hue=labels[region].columns[0])
            plt.show()

    def correlated(how_many: int):
        plot(how_many, True)

    def uncorrelated(how_many: int):
        plot(how_many, False)

    def get_top_most_different(dist, n: int):
        return np.argsort(-np.mean(dist, axis=1).flatten())[:n]

    def get_top_most_different_tuples(dist, n: int):
        return list(zip(*np.unravel_index(np.argsort(-dist.ravel()), dist.shape)))[:n]

    def most_different_features(top_number: int) -> None:
        for region, data in epigenomes.items():
            dist = euclidean_distances(data.T)
            most_distance_columns_indices = get_top_most_different(dist, top_number)
            columns = data.columns[most_distance_columns_indices]
            fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
            print(f"Top {top_number} different features from {region}.")
            for column, axis in zip(columns, axes.flatten()):
                head, tail = data[column].quantile([0.05, 0.95]).values.ravel()

                mask = ((data[column] < tail) & (data[column] > head)).values

                cleared_x = data[column][mask]
                cleared_y = labels[region].values.ravel()[mask]

                cleared_x[cleared_y == 0].hist(ax=axis, bins=20)
                cleared_x[cleared_y == 1].hist(ax=axis, bins=20)

                axis.set_title(column)
            fig.tight_layout()
            plt.show()

    def most_different_tuples(top_number: int) -> None:
        for region, x in epigenomes.items():
            dist = euclidean_distances(x.T)
            dist = np.triu(dist)
            tuples = get_top_most_different_tuples(dist, top_number)
            fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
            print(f"Top {top_number} different tuples of features from {region}.")
            for (i, j), axis in zip(tuples, axes.flatten()):
                column_i = x.columns[i]
                column_j = x.columns[j]
                for column in (column_i, column_j):
                    head, tail = x[column].quantile([0.05, 0.95]).values.ravel()
                    mask = ((x[column] < tail) & (x[column] > head)).values
                    x[column][mask].hist(ax=axis, bins=20, alpha=0.5)
                axis.set_title(f"{column_i} and {column_j}")
            fig.tight_layout()
            plt.show()

    correlated(3)
    uncorrelated(3)
    most_different_features(5)
    most_different_tuples(5)
