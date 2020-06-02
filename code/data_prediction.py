import json
import os
from glob import glob
from typing import List, Dict

import compress_json
import numpy as np
import pandas as pd
from PIL import Image
from barplots import barplots
from sanitize_ml_labels import sanitize_ml_labels
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from tensorboard.notebook import display
from tensorflow.python.keras.callbacks import EarlyStopping
from tqdm import tqdm

from data_retrieval import get_holdout
from defaults import get_default
from meta_models import Model


def _precomputed(results, model: Model, holdout: int) -> bool:
    df = pd.DataFrame(results)
    if df.empty:
        return False
    return (
            (df.model == model.name) &
            (df.holdout == holdout)
    ).any()


def _get_filename() -> str:
    return f"results_{get_default('cell_line')}_{get_default('region')}.json"


def _get_holdouts() -> StratifiedShuffleSplit:
    return StratifiedShuffleSplit(n_splits=get_default('splits'), test_size=get_default('test_size'), random_state=42)


def _report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    integer_metrics = accuracy_score, balanced_accuracy_score
    float_metrics = roc_auc_score, average_precision_score
    results1 = {
        sanitize_ml_labels(metric.__name__): metric(y_true, np.round(y_pred))
        for metric in integer_metrics
    }
    results2 = {
        sanitize_ml_labels(metric.__name__): metric(y_true, y_pred)
        for metric in float_metrics
    }
    return {
        **results1,
        **results2
    }


def _get_result(model: Model, run_type: str, holdout: int, **kwargs) -> Dict:
    return {
        'model': model.name,
        'run_type': run_type,
        'holdout': holdout,
        **kwargs
    }


def predict_epigenomics(data: pd.DataFrame, labels: pd.DataFrame, models: List[Model]) -> List[Dict]:
    if os.path.exists(_get_filename()):
        with open(_get_filename()) as json_file:
            results = json.load(json_file)
    else:
        results = []

    for i, (train, test) in tqdm(enumerate(_get_holdouts().split(data, labels)), total=get_default('splits'),
                                 desc="Computing holdouts", dynamic_ncols=True):
        for model, params in tqdm([model.get_model() for model in models], total=len(models), desc="Training models",
                                  leave=False, dynamic_ncols=True):
            if _precomputed(results, model, i):
                continue
            model.fit(data[train], labels[train], **params)
            results.append(_get_result(model, 'train', i, **_report(labels[train], model.predict(data[train]))))
            results.append(_get_result(model, 'test', i, **_report(labels[test], model.predict(data[test]))))
            compress_json.local_dump(results, _get_filename())
    return results


def predict_sequences(sequences: pd.DataFrame, labels: pd.DataFrame, models: List[Model]) -> List[Dict]:
    if os.path.exists(_get_filename()):
        with open(_get_filename()) as json_file:
            results = json.load(json_file)
    else:
        results = []

    for i, (train_index, test_index) in tqdm(enumerate(_get_holdouts().split(sequences, labels)),
                                             total=get_default('splits'), desc="Computing holdouts",
                                             dynamic_ncols=True):
        train, test = get_holdout(train_index, test_index, sequences, labels)
        for model, params in tqdm([model.get_model() for model in models], total=len(models),
                                  desc="Training models", leave=False, dynamic_ncols=True):
            if _precomputed(results, model.name, i):
                continue
            history = model.model.fit(
                train,
                validation_data=test,
                steps_per_epoch=train.steps_per_epoch,
                validation_steps=test.steps_per_epoch,
                *params,
                callbacks=[
                    EarlyStopping(monitor="val_loss", mode="min", patience=50),
                ]
            ).history
            scores = pd.DataFrame(history).iloc[-1].to_dict()
            results.append(_get_result(model, 'train', i, **_report(labels[train],
                                                                    **{
                                                                        key: value
                                                                        for key, value in scores.items()
                                                                        if not key.startswith("val_")
                                                                    }
                                                                    )))
            results.append(_get_result(model, 'test', i, **_report(labels[test],
                                                                   **{
                                                                       key[4:]: value
                                                                       for key, value in scores.items()
                                                                       if key.startswith("val_")
                                                                   }
                                                                   )))
            compress_json.local_dump(results, _get_filename())
    return results


def show_barplots(results: List[Dict]) -> None:
    df = pd.DataFrame(results)
    df = df.drop(columns=['holdout'])

    barplots(
        df,
        groupby=["model", "run_type"],
        show_legend=False,
        height=5,
        orientation="horizontal"
    )

    for x in glob("barplots/*.png"):
        display(Image.open(x))
