import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import dask.bag as db
import dask.dataframe as dd

FILE_PATH = os.path.dirname(__file__)


class DataProvider:
    """
    Provides test and train data
    1. Load data with dask
    2. Split data
    3. Create new instances with same title and differnet labels if paper has multiple labels
    """

    def __init__(self, train_split: float, experiment_name, experiment_path):
        self.train_split = train_split
        self.test_split = 1 - train_split
        self.experiment_name = experiment_name
        self.experiment_path = experiment_path
        self.label_dict = {}
        self.random_seed = 100

    def get_train_test_split(self) -> Dict:
        """
        Split data in train and test set.

        Returns
        -------
        Dict
        """
        labels, descriptions = self._load_data(self.experiment_name, self.experiment_path)

        data = {}
        valid_data = False
        while not valid_data:
            self.random_seed = np.random.randint(999)

            train_labels, test_labels = labels.random_split([self.train_split, self.test_split],
                                                            random_state=self.random_seed)
            train_descriptions, test_descriptions = descriptions.random_split([self.train_split, self.test_split],
                                                                              random_state=self.random_seed)

            train_descriptions, train_labels = self._split_multiple_research_fields(train_descriptions, train_labels)

            # map research fields to integer labels
            test_labels = test_labels.map(lambda x: [self.label_dict[field] for field in x.split(';')], meta=('x', int))

            test_labels = np.array(test_labels.to_dask_array())
            train_labels = np.array(train_labels.map(self.label_dict).to_dask_array())

            valid_data = self._check_label_consistency(np.unique(test_labels), np.unique(train_labels))
            print(f'Data valid? {valid_data}')

            data = {
                'y_test': test_labels,
                'y_train': train_labels,
                'X_test': test_descriptions,
                'X_train': train_descriptions
            }

        return data

    def _load_data(self, experiment_name: str, exp_path: str) -> Tuple[db.Bag, db.Bag]:
        """
        Load arXiv data from json file with dask.
        Return Labels, descriptions (which consist of title, publisher, abstract) and a Dict of unique labels

        Parameters
        ----------
        exp_path: str
            Path to dataset. Must be json records.
        experiment_name : str
            Name of experiemnt

        Returns
        -------
        Tuple[db.Bag, db.Bag, Dict]
        """
        lines = db.read_text(os.path.join(exp_path, 'arxiv-metadata-distributed_merged.json'))
        records = lines.map(lambda x: json.loads(x))

        if experiment_name == 'title':
            description_bag = records.map(lambda x: x['title'])
        else:
            description_bag = records.map(lambda x: x['title'] + x['abstract'])

        json_path = os.path.join(FILE_PATH, 'arxiv_data/mapping_arxiv_merged_orkg.json')
        with open(json_path, 'r') as infile:
            mapping = json.load(infile)

        # map labels from arxiv to orkg labels
        map_labels = lambda x: ';'.join([mapping[field].strip() for field in x['categories'].split(' ')])
        label_frame = records.map(map_labels).to_dataframe()
        label_frame.columns = ['label']

        label_list = []
        label_combinations = label_frame['label'].unique().compute()
        for label_combi in label_combinations:
            label_list += label_combi.split(';')

        label_set = set(label_list)
        self.label_dict = {field: index for index, field in enumerate(label_set)}

        return label_frame['label'], description_bag.to_dataframe()

    def _split_multiple_research_fields(self, descriptions: dd.DataFrame, labels: dd.Series):
        """
        Create new Instance for each Research Field if Paper has multiple Research Fields.

        Parameters
        ----------
        descriptions : dd.DataFrame
            dask dataframe with all unique descriptions
        labels : dd.Series
            dask Series with multiple labels per instance

        Returns
        -------
        all_data: dd.DataFrame
            dask dataframe with all descriptions, some descriptions appear multiple times
        all_labels: dd.Series
            dask Series with single label per instance
        """
        descriptions.columns = ['text']
        row_list = []

        for label, description in zip(labels.iteritems(), descriptions['text'].iteritems()):

            # keep first label and append others to new pandas frame
            label_names = label[1].split(';')

            for count, name in enumerate(label_names):
                if count == 0:
                    label = (label[0], name)
                else:
                    row_list.append([description[1], name])

        new_frame = pd.DataFrame(row_list, columns=['text', 'label'])

        # append new Series to existing Series
        all_descriptions = descriptions['text'].append(new_frame['text'])
        all_labels = labels.map(lambda x: x.split(';')[0]).append(new_frame['label'])

        return all_descriptions.to_frame(), all_labels

    def _check_label_consistency(self, test_label_set: np.array, train_label_set: np.array) -> bool:
        """
        Check if a label from the test set doesn´t appear in the training set.
        Preprocess test labels which are List of Labels for each instance
        """
        label_list = []
        for labels in test_label_set:
            for label in labels:
                label_list.append(label)
        label_list = set(label_list)

        for test_label in label_list:
            if test_label not in train_label_set:
                return False
        return True
