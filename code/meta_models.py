from multiprocessing import cpu_count
from typing import Tuple, List, Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential

from defaults import get_default


class Model:
    # to avoid warning in other files
    MLP = None
    FFNN = None
    CNN = None

    def __init__(self, name: str, model: Union[Sequential, RandomForestClassifier, DecisionTreeClassifier], **kwargs):
        self.name = name
        self.model = model
        self.kwargs = kwargs

    def __str__(self) -> str:
        """Name of the model."""
        return self.name

    def __repr__(self) -> str:
        """Name of the model."""
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
        def get_model(input_shape: Tuple[int], name: str = None, optimizer: str = get_default('nadam'),
                      loss: str = get_default('loss'), metrics: List = None,
                      epochs: int = get_default('epochs'), batch_size: int = get_default('batch_size'),
                      validation_split: float = get_default('validation_split'), shuffle: bool = True,
                      verbose: bool = get_default('verbose'), **kwargs):
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
                'verbose': verbose
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
