import csv
import json
import os
import string
from typing import List, Dict
import warnings

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

pd.set_option("display.max_columns", None)
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

FILE_PATH = os.path.dirname(__file__)

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')


def text_process(text: str) -> List:
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    4. Remove words

    Parameters
    ----------
    text : str
        string of text
    Returns
    -------
    List
    """
    stemmer = WordNetLemmatizer()

    # remove punctuation and stopwords
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc_nostop = [word.lower() for word in nopunc.split() if word not in stopwords.words('english')]

    return [stemmer.lemmatize(word) for word in nopunc_nostop]


def save_metrics(results: Dict, exp_settings: Dict, exp_path: str):
    """
    Saves Two dicts to json and uses the classifier name and datetime in title.

    Parameters
    ----------
    exp_path : str
        path to experiment
    results : Dict
        Dict with values for Precision, Recall, F1
    exp_settings : Dict
        Dict with hyperparams of Model

    Returns
    -------
    """
    json_path = os.path.join(exp_path, 'meta-info.json')
    with open(json_path, 'w') as fp:
        json.dump({'result': results, 'setting': exp_settings}, fp)


def create_field_table(experiment_path: str):
    """
    Create Table with:
    ResearchField, Arxiv Label, Instance Numbers, Title Precision, Abstract Precision

    Parameters
    ----------
    experiment_path :
        path to experiment folder

    Returns
    -------
    """
    count_path = os.path.join(experiment_path, 'info-arxiv-distribution_all.json')
    with open(count_path, 'r') as jsonfile:
        count_dict = json.load(jsonfile)

    mapping_path = os.path.join(FILE_PATH, 'arxiv_data/mapping_arxiv_orkg.json')
    with open(mapping_path, 'r') as jsonfile:
        mapping_dict = json.load(jsonfile)

    label_path = os.path.join(experiment_path, 'result_precison_singletitle.json')
    with open(label_path, 'r') as jsonfile:
        label_percents_title = json.load(jsonfile)

    label_path = os.path.join(experiment_path, 'result_precison_singletitle_abstract.json')
    with open(label_path, 'r') as jsonfile:
        label_percents_abstract = json.load(jsonfile)

    label_path = os.path.join(experiment_path, 'label-map_title_abstract.json')
    with open(label_path, 'r') as jsonfile:
        label_map = json.load(jsonfile)
        label_map = {v: k for k, v in label_map.items()}

    label_count = {}
    label_labels = {}
    for key, count in count_dict.items():
        orkg_label = mapping_dict.get(key, key)
        label_count[orkg_label] = count if orkg_label not in label_count else label_count[orkg_label] + count
        label_labels[orkg_label] = key if orkg_label not in label_labels else label_labels[orkg_label] + ' ' + key

    with open(os.path.join(experiment_path, "research_field_table.csv"),  "w", newline='') as csv_handler:
        writer = csv.writer(csv_handler)
        writer.writerow(['ORKG Research Field', 'arxiv labels', 'instance number', '% title', '% title + abstract'])

        for key, value in label_count.items():
            arxiv_labels = label_labels[key] if not label_labels[key] == key else '-'
            percentage_title = label_percents_title[int(label_map[key])]
            percentage_abstract = label_percents_abstract[int(label_map[key])]

            writer.writerow([key, arxiv_labels, value, percentage_title, percentage_abstract])
