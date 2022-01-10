from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class AbstractEmbedding(ABC):

    @abstractmethod
    def get_trainining_data(self) -> np.ndarray:
        """
        Returns iterable of numpy arrays.
        Batch size defines how many training isntances are included in each numpy array.

        Returns
        -------
        np.ndarray
            iterable class that provides training data in np.array batches
            for sklearn tfidf it only provides one batch with the whole data.
        """
        pass

    @abstractmethod
    def get_test_data(self) -> np.ndarray:
        """
        Returns iterable of numpy arrays.
        Batch size defines how many training isntances are included in each numpy array.

        Returns
        -------
        np.ndarray
            iterable class that provides training data in np.array batches
            for sklearn tfidf it only provides one batch with the whole data.
        """
        raise NotImplementedError

    @abstractmethod
    def set_data(self, data: Dict):
        """ set data dict """
        raise NotImplementedError

    @abstractmethod
    def calculate_data_embedding(self):
        """
        Calculates new embedding.

        Returns
        -------
        """
        raise NotImplementedError
