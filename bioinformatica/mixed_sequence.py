from typing import Dict, Union, Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence


def sequence_length(sequence: Union[List, np.ndarray], batch_size: int) -> int:
    """Return number of batch sizes contained in sequence.
    Parameters
    -----------
    sequence: List,
        Iterable to split into batches.
    batch_size: int,
        Size of the batches.
    Returns
    -----------
    Return number of batch size contained in given sequence, by excess.
    """

    return int(np.ceil(len(sequence) / float(batch_size)))


def batch_slice(index: int, batch_size: int) -> slice:
    """Return slice corresponding to given index for given batch_size.
    Parameters
    ---------------
    index: int,
        Index corresponding to batch to be rendered.
    batch_size: int
        Batch size for the current Sequence.
    Returns
    ---------------
    Return slice corresponding to given index for given batch_size.
    """

    return slice(index * batch_size, (index + 1) * batch_size)


class NumpySequence(Sequence):
    """NumpySequence is a Sequence wrapper to uniform Numpy Arrays as Keras Sequences.
    Usage Examples
    ----------------------------
    The main usage of this class is as a package private wrapper for Sequences.
    It is required to uniformely return a batch of the array,
    without introducing special cases.
    However, a basic usage example could be the following:
    Wrapping a numpy array as a Sequence
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. code:: python
        from keras_mixed_sequence import NumpySequence
        import numpy as np
        examples_number = 1000
        features_number = 10
        batch_size = 32
        my_array = np.random.randint(
            2, shape=(
                examples_number,
                features_number
            )
        )
        my_sequence = NumpySequence(my_array, batch_size)
        # Keras will require the i-th batch as follows:
        ith_batch = my_sequence[i]
    """

    def __init__(
            self,
            array: np.ndarray,
            batch_size: int,
            seed: int = 42,
            elapsed_epochs: int = 0,
            dtype=float
    ):
        """Return new NumpySequence object.
        Parameters
        --------------
        array: np.ndarray,
            Numpy array to be split into batches.
        batch_size: int,
            Batch size for the current Sequence.
        seed: int = 42,
            Starting seed to use if shuffling the dataset.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        dtype = float,
            Type to which to cast the array if it is not already.
        Returns
        --------------
        Return new NumpySequence object.
        """

        if array.dtype != dtype:
            array = array.astype(dtype)

        self._array, self._batch_size = array, batch_size
        self._seed, self._elapsed_epochs = seed, elapsed_epochs

    def on_epoch_end(self):
        """Shuffle private numpy array on every epoch end."""

        state = np.random.RandomState(seed=self._seed + self._elapsed_epochs)

        self._elapsed_epochs += 1

        state.shuffle(self._array)

    def __len__(self) -> int:
        """Return length of Sequence."""

        return sequence_length(
            self._array,
            self._batch_size

        )

    def __getitem__(self, idx: int) -> np.ndarray:
        """Return batch corresponding to given index.
        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be rendered.
        Returns
        ---------------
        Return numpy array corresponding to given batch index.
        """

        return self._array[batch_slice(idx, self._batch_size)]


class MixedSequence(Sequence):
    """Handles Mixed type input / output Sequences.
    Usage examples
    -----------------------------
    """

    def __init__(

            self,

            x: Union[Dict[str, Union[np.ndarray, Sequence]], np.ndarray, Sequence],

            y: Union[Dict[str, Union[np.ndarray, Sequence]], np.ndarray, Sequence],

            batch_size: int

    ):

        # Casting to dictionary if not one already

        x, y = [

            e if isinstance(e, Dict) else {0: e}

            for e in (x, y)

        ]

        # Retrieving sequence length

        self._sequence_length = None

        self._batch_size = batch_size

        for candidate in (*x.values(), *y.values()):

            if isinstance(candidate, Sequence):
                self._sequence_length = len(candidate)

                break

        # Veryfing that at least a sequence was provided

        if self._sequence_length is None:
            raise ValueError("No Sequence was provided.")

        # Converting numpy arrays to Numpy Sequences

        x, y = [

            {

                key: NumpySequence(candidate, batch_size) if isinstance(

                    candidate, np.ndarray) else candidate

                for key, candidate in dictionary.items()

            }

            for dictionary in (x, y)

        ]

        # Checking that every value within the dictionaries

        # is now a sequence with the same length.

        for dictionary in (x, y):

            for _, value in dictionary.items():

                if len(self) != len(value):
                    raise ValueError(

                        "One or given sub-Sequence does not match length of other Sequences."

                    )

        self._x, self._y = x, y

    def on_epoch_end(self):
        """Call on_epoch_end callback on every sub-sequence."""

        for dictionary in (self._x, self._y):

            for _, value in dictionary.items():
                value.on_epoch_end()

    def __len__(self) -> int:
        """Return length of Sequence."""

        return self._sequence_length

    @property
    def steps_per_epoch(self) -> int:
        """Return length of Sequence."""

        return len(self)

    def __getitem__(self, idx: int) -> Tuple[

        Union[np.ndarray, Dict],

        Union[np.ndarray, Dict]

    ]:

        """Return batch corresponding to given index.
        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be rendered.
        Returns
        ---------------
        Return Tuple containing input and output batches.
        """

        return tuple([
                         {
                             key: sequence[idx]
                             for key, sequence in dictionary.items()
                         } if len(dictionary) > 1 else next(iter(dictionary.values()))[idx]
                         for dictionary in [
                self._x,
                self._y
            ]
                     ] + (
                         []
                         if tf.__version__.startswith("1.14")
                         else
                         [{key: None for key in self._y}]
                     ))
