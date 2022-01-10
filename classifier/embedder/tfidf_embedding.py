import os
from typing import Any, Dict
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from classifier.embedder.abstract_embedding import AbstractEmbedding
from classifier.iterators.tfidf_iterator import TfidfIterator
from classifier.iterators.data_iterator import DataIterator
from classifier.classify_util import text_process
from logs.my_logger import MyLogger
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

logger = MyLogger('tfidf_matrix').logger
FILE_PATH = os.path.dirname(__file__)


class TfidfMatrix(AbstractEmbedding):

    def __init__(self, exp_name: str,
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
        self.tfidf = TfidfModel()

        self.data = {
            'y_test': Any,
            'y_train': Any,
            'X_test': TfidfIterator,
            'X_train': TfidfIterator,
        }

    def get_trainining_data(self) -> DataIterator:
        """
        Returns iterable of numpy arrays.
        Batch size defines how many training isntances are included in each numpy array.

        Returns
        -------
        DataIterator
            iterable class that provides training data in batches
        """
        return DataIterator(self.batch_size, self.vocab_size, self.tfidf, self.data['X_train'], self.data['y_train'])

    def get_test_data(self):
        """
        Returns iterable of numpy arrays.
        Batch size defines how many training isntances are included in each numpy array.

        Returns
        -------
        TestDataIterator
        """
        return DataIterator(self.batch_size, self.vocab_size, self.tfidf, self.data['X_test'], self.data['y_test'])

    def set_data(self, data: Dict):
        """ set data dict """
        assert('y_test' in data and 'y_train' in data and 'X_train' in data and 'X_test' in data)
        self.data = data

    def calculate_data_embedding(self):
        """
        Loads data as dask bags from load_orkg_data.
        Calculates new tfidf_model or loads existing one from disk.
        Take in mind that whole arxiv data is approx. 3 GB json.

        Returns
        -------
        Dict
            Mapping of Research Fields to integer
        """
        self._calculate_tf_idf_matrix()
        self._save_tfidf_model()
        logger.info('Embedder loaded.')

    def _calculate_tf_idf_matrix(self) -> None:
        """
        Calculates the tfidf matrix from description text in dataframe.
        Columns are tokens and rows are the descriptions.

        Returns
        -------
        """
        # preprocess data
        train_dd = self.data['X_train'].map(lambda text: text_process(text)).map(lambda s: ' '.join(s))
        test_dd = self.data['X_test'].map(lambda text: text_process(text)).map(lambda s: ' '.join(s))

        # build corpus for Tfidf while iterating over train and test
        train_corpus_iterator = TfidfIterator(train_dd, dictionary=Dictionary())
        test_corpus_iterator = TfidfIterator(test_dd, dictionary=Dictionary())

        # uses gensim library to compute tfidf vectors and convert to sparse matrix
        self.tfidf = TfidfModel(train_corpus_iterator)

        self.data['X_train'], self.data['X_test'] = train_corpus_iterator, test_corpus_iterator
        self.vocab_size = len(train_corpus_iterator.dictionary)

    def _save_tfidf_model(self) -> None:
        """ Saves tfidf model to disk in models folder. """
        model_path = os.path.join(self.experiment_path, "embedding_tfidf_gensim_" + self.experiment_name + ".pkl")
        self.tfidf.save(model_path)
