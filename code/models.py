from typing import Tuple

from tensorflow.python import Reshape
from tensorflow.python.keras.layers import Dense, BatchNormalization, Activation, Dropout, ThresholdedReLU, \
    AlphaDropout, Flatten, Conv2D

from meta_models import Model


def get_mlp_egigenomics():
    return Model.MLP(
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
    )


def get_ffnn_epigenomics_v1():
    return Model.FFNN(
        Dense(256, activation="relu"),
        Dense(128),
        BatchNormalization(),
        Activation("relu"),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
    )


def get_ffnn_epigenomics_v2():
    return Model.FFNN(
        Dense(256, activation="relu"),
        Dense(128),
        BatchNormalization(),
        ThresholdedReLU(0.05),
        Dense(64, activation="relu"),
        AlphaDropout(0.3),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
    )


def get_ffnn_epigenomics_v3():
    return Model.FFNN(
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        BatchNormalization(),
        ThresholdedReLU(0.05),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        AlphaDropout(0.5),  # new
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
    )


def get_mlp_sequential():
    return Model.MLP(
        Flatten()
    )


def get_ffnn_sequential():
    return Model.FFNN(
        Flatten(),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
    )


def get_cnn_sequential_v1():
    return Model.CNN(
        Reshape((200, 4, 1)),
        Conv2D(64, kernel_size=(10, 2), activation="relu"),
        Conv2D(64, kernel_size=(10, 2), activation="relu"),
        Dropout(0.3),
        Conv2D(32, kernel_size=(10, 2), strides=(2, 1), activation="relu"),
        Conv2D(32, kernel_size=(10, 1), activation="relu"),
        Conv2D(32, kernel_size=(10, 1), activation="relu"),
        Dropout(0.3),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
    )
