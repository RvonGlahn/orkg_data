import numpy as np
import os
from typing import Tuple
from tqdm import tqdm
from gensim.models import TfidfModel
from classifier.iterators import tfidf_iterator
import math

FILE_PATH = os.path.dirname(__file__)


class DataIterator:
    """
    Generates batches of numpy arrays.
    Create iteratively numpy arrays from tfidf model.
    """

    def __init__(
            self,
            batch_size: int,
            vocab_length: int,
            model: TfidfModel,
            doc_iterator: tfidf_iterator,
            labels: np.ndarray
    ):
        """
        Parameters
        ----------
        batch_size :
            number of instances that get created per batch
        vocab_length :
            lenght of embedding array
        model:
            tfidf model of whole corpus
        doc_iterator:
            iterable over all documents in dataset
        """
        self.batch_size = batch_size
        self.vocab_length = vocab_length
        self.model = model
        self.iterator = doc_iterator
        self.label_array = labels

    def __iter__(self) -> np.ndarray:
        """
        Creates from model batch numpy arrays with embedding and  an array of labels

        Returns
        -------
        np.ndarray
            Batch of tfidf embeddings
        """
        count = 0
        batch_labels = np.zeros(self.batch_size)
        batch_array = np.empty((self.batch_size, self.vocab_length))

        with tqdm(total=math.ceil(len(self.label_array)/self.batch_size)) as pbar:
            for document in self.model[self.iterator]:

                if not count % self.batch_size and count:
                    pbar.update(1)
                    yield {'data': batch_array, 'label': batch_labels}

                # update array
                batch_array[count % self.batch_size, :] = self._build_array_from_tfidf(document)
                batch_labels[count % self.batch_size] = self.label_array[count]
                count += 1

            pbar.update(1)
            final_count = count % self.batch_size
            yield {'data': batch_array[0:final_count][:], 'label': batch_labels[0:final_count]}

    def _build_array_from_tfidf(self, document: Tuple[int, float]) -> np.ndarray:
        """
        Build numpy array from tfidf dictionary

        Parameters
        ----------
        document : Tuple[int, float]
            Tuple with int index and float value for tfidf

        Returns
        -------
        np.ndarray
            Tfidf array of single document with whole corpus length
        """
        array = np.zeros(self.vocab_length)

        for index, value in document:
            array[index] = value

        return array
