import unittest
from deployment.research_field_classifier import ResearchFieldClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


class TestResearchFieldClassifier(unittest.TestCase):

    def test_load_embedding(self):
        classifier = ResearchFieldClassifier()
        self.assertTrue(hasattr(classifier.abstract_path, 'embedder'))
        self.assertTrue(hasattr(classifier.title_path, 'embedder'))

        tfidf = classifier._load_embedding(classifier.title_path['embedder'])
        self.assertTrue(type(tfidf) == TfidfVectorizer)
        tfidf = classifier._load_embedding(classifier.abstract_path['embedder'])
        self.assertTrue(type(tfidf) == TfidfVectorizer)

    def test_load_model(self):
        classifier = ResearchFieldClassifier()
        self.assertTrue(hasattr(classifier.abstract_path, 'model'))

        model = classifier._load_model(classifier.title_path['model'])
        self.assertTrue(type(model) == SGDClassifier)
        model = classifier._load_model(classifier.abstract_path['model'])
        self.assertTrue(type(model) == SGDClassifier)

    def test_load_labels(self):
        classifier = ResearchFieldClassifier()
        self.assertTrue(hasattr(classifier.title_path, 'label'))
        labels = classifier._load_labels(classifier.title_path['label'])
        self.assertTrue(type(labels) == dict)

    def test_predict_field(self):
        title = 'A Neural Conversation Generation Model via Equivalent Shared Memory Investigation'
        abstract = 'Conversation generation as a challenging task in Natural Language Generation (NLG)'

        classifier = ResearchFieldClassifier()
        self.assertTrue(type(classifier.predict_research_field(title)) == str)
        self.assertTrue(type(classifier.predict_research_field(title, abstract)) == str)
