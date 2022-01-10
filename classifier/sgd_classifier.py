import json
import pickle
import os
from statistics import mean
from typing import Dict
import warnings

import numpy as np
from ray.tune.sklearn import TuneSearchCV
from sklearn.linear_model import SGDClassifier

from classifier.data_provider import DataProvider
from classifier.metrics import custom_f1, custom_precision, custom_recall
from classifier.embedder.tfidf_embedding import TfidfMatrix
from classifier.embedder.sentence_embedding import SentenceEmbedding
from classifier.embedder.tfidf_sklearn_embedding import Tfidf_Sklearn_Embedding
from logs.my_logger import MyLogger

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
logger = MyLogger('sgd_classifier').logger
FILE_PATH = os.path.dirname(__file__)

SGD_PARAMS = {
    "loss": 'log',
    "max_iter": 1000,
    "n_jobs": 4,
    "alpha": 1e-7,
    "tol": 1e-6,
}

TUNE_GRID_PARAMS = {
    "tol": (1e-7, 1e-2),
    "alpha": (1e-8, 0.1),
}

TUNE_SETTINGS = {
    'iterations': 20,
    'stopping': True,
    'trials': 4
}


class MySGDClassifier:

    def __init__(
            self,
            experiment_path: str,
            embedding_type: str = 'tfidf',
            batch_size: int = 1000000,
            auto_opt: bool = False,
            train_split: float = 0.7,
    ):
        """
        Initialzes Parameters of Classifier and Parameters of the Optimizer for every partial fit
        Sets Classifier to SGDClassifier.
        Initialize Embedder.

        Parameters
        ----------
        experiment_path: str
            path to experiment_folder
        embedding_type : str
            string that decides which embedder will be used
        batch_size: int
            batch size, but no really used
        auto_opt: bool,
            if True HPO is used
        train_split:
            percentage of training split
        """
        self.sgd_model = SGDClassifier(**SGD_PARAMS)
        self.tune_model: TuneSearchCV = TuneSearchCV
        self.auto_opt = auto_opt

        # parameters for SGD classifier
        self.tune_parameter_grid = TUNE_GRID_PARAMS
        self.tune_setting = TUNE_SETTINGS
        self.label_dict = {}
        self.experiment_name = 'title'
        self.experiment_path = experiment_path
        self.embed_name = embedding_type

        # data processing
        self.random_seed = 100
        self.data_loader = DataProvider(train_split, self.experiment_name, experiment_path)

        # embedders
        if embedding_type == 'tfidf':
            self.embedding = TfidfMatrix(self.experiment_name, experiment_path, batch_size)
        elif embedding_type == 'tfidf sklearn':
            self.embedding = Tfidf_Sklearn_Embedding(self.experiment_name, experiment_path, batch_size)
        elif embedding_type == 'sentence embedding':
            self.embedding = SentenceEmbedding(self.experiment_name, experiment_path, batch_size)
        else:
            ValueError('Wrong Embedder selected. Check your Inputs and try again')

    def set_experiment_name(self, experiment_name: str):
        """ set experiment name """
        self.experiment_name = experiment_name
        self.data_loader.experiment_name = experiment_name
        self.embedding.experiment_name = experiment_name

    def run_experiment(self, dataset_name: str = 'arxiv') -> Dict:
        """
        Runs experiment and returns metrics in dict.
        Expects iterable for batch processing (Class that has __iter__ method implemented)

        Parameters
        ----------
        dataset_name: str
            string that identifies dataset

        Returns
        -------
        Dict
        """
        logger.info(f'Experiment "{self.experiment_name}" started on {dataset_name}-dataset.')

        self.embedding.set_data(self.data_loader.get_train_test_split())
        self.embedding.calculate_data_embedding()

        self.label_dict = dict((v, k) for k, v in self.data_loader.label_dict.items())

        # train model
        for train_batch in self.embedding.get_trainining_data():
            if self.auto_opt:
                self._train_optimal_sgd(train_batch['data'], train_batch['label'])
            else:
                self._train_sgd(train_batch['data'], train_batch['label'])

        self._save_model()

        # predict labels
        result_list = {'custom_precision': [], 'custom_recall': [], 'custom_f1 score': []}
        for test_batch in self.embedding.get_test_data():
            result_batch = self._predict_sgd(test_batch['data'], test_batch['label'])

            for metric_name, metric_score in result_batch.items():
                result_list[metric_name].append(metric_score)

        result = {'custom_precision': mean(result_list['custom_precision']),
                  'custom_recall': mean(result_list['custom_recall']),
                  'custom_f1 score': mean(result_list['custom_f1 score'])}

        logger.info(result)
        return result

    def _train_optimal_sgd(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the SGD Model partially batchwise.

        Parameters
        ----------
        X_train: np.ndarray
            train embeddings batch
        y_train: np.ndarray
            train labels batch
        """
        self.tune_model = TuneSearchCV(
            SGDClassifier(),
            self.tune_parameter_grid,
            n_jobs=1,
            search_optimization="bohb",
            n_trials=self.tune_setting['trials'],
            early_stopping=self.tune_setting['stopping'],
            max_iters=self.tune_setting['iterations'],
        )

        self.tune_model.fit(X_train, y_train)
        self.sgd_model = self.tune_model
        logger.info(f"Best Optimal Params: {self.tune_model.best_params_}")

    def _train_sgd(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the SGD Model in one Go.

        Parameters
        ----------
        X_train: np.ndarray
            train embeddings batch
        y_train: np.ndarray
            train labels batch
        """
        self.sgd_model.fit(X_train, y_train)

    def _predict_sgd(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Makes predictions and calculates metrics: f1-score, precison, recall

        Parameters
        ----------
       X_test: np.ndarray
            test embeddings
        y_test: np.ndarray
            test labels

        Returns
        -------
        Dict
        """
        # loop over test data
        prediction = self.sgd_model.predict(X_test)

        metric_scores = {
            'custom_precision': custom_precision(y_test, prediction, self.experiment_path, self.experiment_name),
            'custom_recall': custom_recall(y_test, prediction),
            'custom_f1 score': custom_f1(y_test, prediction),
        }

        return metric_scores

    def _save_model(self):
        """ Save model to disk """
        path = os.path.join(self.experiment_path, 'model_sgd_' + self.experiment_name + '.pkl')

        with open(path, 'wb') as f:
            pickle.dump(self.sgd_model, f)

        info_path = os.path.join(self.experiment_path, 'label-map_' + self.experiment_name + '.json')
        with open(info_path, "w") as fp:
            json.dump(self.label_dict, fp)

    def _load_model(self, filename: str):
        """ Load model from experiment directory """
        path = os.path.join(self.experiment_path, filename)

        with open(path, 'rb') as f:
            self.sgd_model = pickle.load(f)
