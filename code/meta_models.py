from typing import Tuple, List, Union
from multiprocessing import cpu_count

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential


class Model:
    def __init__(self, name: str, model: Union[Sequential, RandomForestClassifier, DecisionTreeClassifier], **kwargs):
        self.name = name
        self.model = model
        self.kwargs = kwargs

    def __getattr__(self, item):
        print(item)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def get_model(self) -> Tuple:
        return self.model, self.kwargs


def _get_decision_tree(default_name: str = 'DecisionTree', criterion: str = 'gini', max_depth: int = 50,
                       random_state: int = 42, class_weight: str = 'balanced'):
    def get_model(name: str = None, **kwargs):
        name = name or default_name
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight
        )
        return Model(name, model, **kwargs)
    return get_model


Model.DecisionTree = _get_decision_tree()


def _get_random_forest(default_name: str = 'RandomForest', n_estimators: int = 500, criterion: str = 'gini',
                       max_depth: int = 30, random_state: int = 42,
                       class_weight: str = 'balanced', n_jobs: int = cpu_count):
    def get_model(name: str = None, **kwargs):
        name = name or default_name
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=n_jobs
        )
        return Model(name, model, **kwargs)
    return get_model


Model.RandomForest = _get_random_forest()


def _get_sequential(default_name: str = 'Sequential'):
    # no first and last layer
    def get_layers(*hidden_layers: Tuple[Layer]):
        def get_model(input_shape: Tuple[int], name: str = None, optimizer: str = 'nadam',
                      loss: str = 'binary_crossentropy', metrics: List = None,
                      epochs: int = 1000, batch_size: int = 1024,
                      validation_split: float = 0.1, shuffle: bool = True, verbose: bool = False,
                      callbacks: List = None, **kwargs):
            name = name or default_name
            input_layers = (Input(shape=input_shape),)
            output_layers = (Dense(1, activation="sigmoid"),)
            layers = (*input_layers, *hidden_layers, *output_layers)
            model = Sequential(layers, name)

            metrics = metrics or [
                "accuracy",
                AUC(curve="ROC", name="auroc"),
                AUC(curve="PR", name="auprc")
            ]
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )

            kwargs.update({
                'epochs': epochs,
                'batch_size': batch_size,
                'validation_split': validation_split,
                'shuffle': shuffle,
                'verbose': verbose,
                'callbacks': callbacks
            })
            model.summary()
            return Model(name, model, **kwargs)

        return get_model

    return get_layers


Model.Sequential = _get_sequential()
Model.Perceptron = _get_sequential('Perceptron')()
Model.MLP = _get_sequential('MLP')
Model.FFNN = _get_sequential('FFNN')
Model.CNN = _get_sequential('CNN')
