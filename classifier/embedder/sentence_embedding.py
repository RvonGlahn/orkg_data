import os
import numpy as np
from typing import List, Tuple, Any, Dict
import tensorflow_hub as hub
from logs.my_logger import MyLogger
from classifier.embedder.abstract_embedding import AbstractEmbedding
from tqdm import tqdm
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logger = MyLogger('sentence_embedding').logger
FILE_PATH = os.path.dirname(__file__)

SENTENCE_MAX = {
    'title': 1,
    'title_publisher': 2,
    'title_abstract': 10,
}


class SentenceEmbedding(AbstractEmbedding):

    def __init__(self, exp_name: str,
                 experiment_path: str,
                 batch_size: int = 1000
                 ):
        """
        Initialize Embedding with experiment name.
        Uses Universal Senetence Enocder for Embedding.

        Parameters
        ----------
        exp_name : str
            experiment name that determines the infos used for classififcation later in the model.
        experiment_path: str
            path to experiment folder
        batch_size: int
            Keep batch size as big as possible
        """
        self.experiment_name = exp_name
        self.experiment_path = experiment_path
        self.batch_size = batch_size
        self.embedding_length = SENTENCE_MAX[exp_name]
        self.embedder = hub.load(os.path.join(FILE_PATH, 'models', 'hub'))

        # data variables
        self.data = {
            'y_test': Any,
            'y_train': Any,
            'X_test': Any,
            'X_train': Any,
        }

    def get_trainining_data(self):
        """ Call data_iterator to get training data """
        return self._data_iterator('train')

    def get_test_data(self):
        """ Call data_iterator to get test data """
        return self._data_iterator('test')

    def set_data(self, data: Dict):
        """ set data dict """
        assert('y_test' in data and 'y_train' in data and 'X_train' in data and 'X_test' in data)
        self.data = data

    def calculate_data_embedding(self):
        """
        Loads data as dask bags from load_arxiv_data.
        Stores data in object varibales.
        Take in mind that whole arxiv data is approx. 3 GB json.

        Returns
        -------
        Dict
            Mapping of Research Fields to integer
        """
        logger.info('Data loaded.')

    def _data_iterator(self, train_type: str):
        """
        Iterates over train or test data an yields embedded numpy arrays for classification.
        Show progress bar for processed batches.

        Parameters
        ----------
        train_type : str
            test or train.

        Returns
        -------
        """
        data = self.data['X_train'] if train_type == 'train' else self.data['X_test']
        labels = self.data['y_train'] if train_type == 'train' else self.data['y_test']

        count = 0
        batch_labels = np.empty(self.batch_size) if train_type == 'train' else []

        sentence_list = []

        with tqdm(total=math.ceil(len(labels) / self.batch_size)) as pbar:
            for line in data.itertuples():

                if not count % self.batch_size and count:
                    batch_array = self._calculate_embeddings(sentence_list)
                    sentence_list = []
                    pbar.update(1)

                    yield {'data': batch_array, 'label': batch_labels}
                    batch_labels = np.empty(self.batch_size) if train_type == 'train' else []

                # update array
                sentence_list.append(line)

                if train_type == 'train':
                    batch_labels[count % self.batch_size] = labels[count]
                else:
                    batch_labels.append(labels[count])
                count += 1

            pbar.update(1)
            final_count = count % self.batch_size
            batch_array = self._calculate_embeddings(sentence_list)
            yield {'data': batch_array[0:final_count][:], 'label': batch_labels[0:final_count]}

    def _calculate_embeddings(self, sentence_list: List[str]) -> np.ndarray:
        """
        Calculates embeddings with universal sentence encoder from tensorflow. Each sentence ist a (512,) numpy array
        Uses padding to get output of same length for multi sentence descriptions
        Uses clipping only for test data to get output of same length for multi sentence descriptions

        Parameters
        ----------
        sentence_list: List[str]
            List of descriptions for all training instances

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
        """
        embeddings = self._sentence_to_embedding(sentence_list)
        embeddings = self._add_padding_clipping(embeddings)

        return np.vstack(np.array(embeddings, dtype=np.ndarray))

    def _sentence_to_embedding(self,
                               string_list: List[str],
                               ) -> List[np.ndarray]:
        """
        Transforms strings of descriptions in embedded sentences and calculates embeddings
        with universal sentence encoder from tf hub. Each sentence ist a (512,) numpy array.

        Parameters
        ----------
        string_list: List[str]
            List of all descriptions

        Returns
        -------
        Tuple[List[np.ndarray], int]
        """
        embeddings = []
        for description_string in string_list:
            embed_text = []
            description_string = description_string[1]
            # split strings to list of sentences to fit model input requirements
            if 'abstract' in self.experiment_name:
                [embed_text.append(sentence) for sentence in description_string.split('.') if len(sentence) > 5]
            else:
                embed_text.append(description_string)

            embeddings.append(np.array(self.embedder(embed_text)))

        return embeddings

    def _add_padding_clipping(self, embeddings: List[np.ndarray]) -> List[np.ndarray]:
        """
        Adds padding to smaller embeddings and uses clipping for too long embeddings
        to create same sized flattened array

        Parameters
        ----------
        embeddings: List[np.ndarray]
            List of all embeddings that have differnet size

        Returns
        -------
        List[np.ndarray]
        """
        for index, test_embedding in enumerate(embeddings):
            embed_size = np.shape(test_embedding)[0]

            while embed_size > self.embedding_length:
                test_embedding = np.delete(test_embedding, -1, axis=0)
                embed_size -= 1

            if embed_size < self.embedding_length:
                padded_array = np.concatenate((test_embedding, np.zeros((self.embedding_length - embed_size, 512))))
            else:
                padded_array = test_embedding

            embeddings[index] = padded_array.flatten()

        return embeddings
