import os
import pickle
import warnings
from typing import Any, Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from classifier.embedder.abstract_embedding import AbstractEmbedding
from classifier.classify_util import text_process
from logs.my_logger import MyLogger

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

logger = MyLogger('tfidf_matrix').logger
FILE_PATH = os.path.dirname(__file__)


class Tfidf_Sklearn_Embedding(AbstractEmbedding):

    def __init__(self,
                 exp_name: str,
                 experiment_path: str,
                 batch_size: int = 1000,
                 ):
        """
        Parameters
        ----------
        exp_name : str
            name of experiment (title, title_abstract ...)
        experiment_path: str
            path to experiment folder
        batch_size : int
            Keep this as big as possible
        """
        self.experiment_name = exp_name
        self.experiment_path = experiment_path
        self.batch_size = batch_size
        self.vocab_size = 0
        self.tfidf = TfidfVectorizer()

        self.data = {
            'y_test': Any,
            'y_train': Any,
            'X_test': Any,
            'X_train': Any,
        }

    def get_trainining_data(self):
        """ Call data_iterator to get training data """
        return [{'data': self.data['X_train'], 'label': self.data['y_train']}]

    def get_test_data(self):
        """ Call data_iterator to get test data """
        return [{'data': self.data['X_test'], 'label': self.data['y_test']}]

    def set_data(self, data: Dict):
        """ set data dict """
        assert('y_test' in data and 'y_train' in data and 'X_train' in data and 'X_test' in data)
        self.data = data

    def calculate_data_embedding(self):
        """
        Calculates new tfidf_model.
        """
        self._calculate_tf_idf_matrix()
        logger.info('Model Arxiv loaded.')

    def _calculate_tf_idf_matrix(self) -> None:
        """
        Calculates the tfidf matrix from description text in dataframe.
        Columns are tokens and rows are the descriptions.

        Returns
        -------
        """
        train_df = pd.DataFrame(self.data['X_train'].compute())
        test_df = pd.DataFrame(self.data['X_test'].compute())

        train_df.columns = ['text']
        test_df.columns = ['text']

        train_df['Text preprocessed'] = train_df['text'].apply(text_process)
        train_df['Text preprocessed str'] = train_df['Text preprocessed'].apply(' '.join)

        test_df['Text preprocessed'] = test_df['text'].apply(text_process)
        test_df['Text preprocessed str'] = test_df['Text preprocessed'].apply(' '.join)

        tfidfconvert = TfidfVectorizer(ngram_range=(1, 2))

        train_tfidf = tfidfconvert.fit_transform(train_df['Text preprocessed str'])
        test_tfidf = tfidfconvert.transform(test_df['Text preprocessed str'])

        self.data['X_train'], self.data['X_test'] = train_tfidf, test_tfidf

        self._save_tfidf_model(tfidfconvert)

    def _save_tfidf_model(self, model) -> None:
        """ Saves tfidf model to disk in models folder. """
        model_path = os.path.join(self.experiment_path, "embedding_tfidf_" + self.experiment_name + ".pkl")

        with open(model_path, 'wb') as fh:
            pickle.dump(model, fh)
