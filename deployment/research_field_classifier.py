import argparse
import json
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

FILE_PATH = os.path.dirname(__file__)


class ResearchFieldClassifier:
    """
    Classifies papers with the help of title and optional the abstract.
    Uses tfidf model for text embedding and SGD for classification.
    """

    def __init__(self):
        """ set up path and loads model, embedder and labels"""
        self.abstract_path = {
            'embedder': os.path.join(FILE_PATH, 'data/embedding_tfidf_title_abstract.pkl'),
            'model': os.path.join(FILE_PATH, 'data/model_sgd_title_abstract.pkl'),
            'label': os.path.join(FILE_PATH, 'data/label-map_title_abstract.json')
        }
        self.title_path = {
            'embedder': os.path.join(FILE_PATH, 'data/embedding_tfidf_title.pkl.pkl'),
            'model': os.path.join(FILE_PATH, 'data/model_sgd_title.pkl.pkl'),
            'label': os.path.join(FILE_PATH, 'data/label-map_title.json.json')
        }

        self.abstract_embedder: TfidfVectorizer = self._load_embedding(self.abstract_path['embedder'])
        self.abstract_model: SGDClassifier = self._load_model(self.abstract_path['model'])
        self.abstract_label_mapping: dict = self._load_labels(self.abstract_path['label'])

        self.title_embedder: TfidfVectorizer = self._load_embedding(self.title_path['embedder'])
        self.title_model: SGDClassifier = self._load_model(self.title_path['model'])
        self.title_label_mapping: dict = self._load_labels(self.title_path['label'])

    def _load_embedding(self, path) -> TfidfVectorizer:
        """ load embedder from pickle file """
        with open(path, "rb") as embed_file:
            tfidftransformer = pickle.load(embed_file)
        return tfidftransformer

    def _load_model(self, path) -> SGDClassifier:
        """ load model from pickle file """
        with open(path, 'rb') as f:
            sgd_model = pickle.load(f)
        return sgd_model

    def _load_labels(self, path) -> dict:
        """ load labels from pickle file """
        with open(path, 'r') as jsonfile:
            label_mapping = json.load(jsonfile)
        return label_mapping

    def predict_research_field(self, title: str, abstract: str = '') -> str:
        """
        Predicts ORKG Research Field based on title and abstract.

        Parameters
        ----------
        title : str
            paper title
        abstract : str, optional
            paper abstract

        Returns
        -------
        str
            ORKG Research Field
        """
        if abstract:
            text = [title + abstract]
            embedded_text = self.abstract_embedder.transform(text)
            prediction = str(self.abstract_model.predict(embedded_text)[0])
            label = self.abstract_label_mapping[prediction]

        else:
            text = [title]
            embedded_text = self.title_embedder.transform(text)
            prediction = str(self.title_model.predict(embedded_text)[0])
            label = self.title_label_mapping[prediction]

        return label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', type=str, help='paper title')
    parser.add_argument('--abstract', type=str, help='paper abstract', default='')
    args = parser.parse_args()

    classifier = ResearchFieldClassifier()
    classifier.predict_research_field(args.title, args.abstract)
