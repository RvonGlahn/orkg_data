import json
import numpy as np
import os
import pandas as pd
import random
from typing import List, Dict, Tuple

from sklearn.utils import shuffle

FILE_PATH = os.path.dirname(__file__)


class ProcessArxivData:
    """
    Process Arxiv Data and add new Data.
    """

    def __init__(self):
        self.relevant_columns = ['title', 'abstract', 'categories', 'doi']
        self.arxiv_df = self._load_arxiv_data()
        self.mapping_arxiv_orkg = self._load_arxiv_orkg_mapping()
        self.arxiv_labels = list(self.mapping_arxiv_orkg.keys())
        self.arxiv_distribution = {}
        self.arxiv_distribution_reduced = {}

    def save_df_as_json(self, path):
        """ shuffle data and save it to disk as json records """
        df = shuffle(self.arxiv_df)
        df = df.dropna(subset=['title', 'abstract'])
        df.reset_index(inplace=True, drop=True)

        reduce_path = os.path.join(path, 'arxiv-metadata-distributed_merged.json')
        df.to_json(reduce_path, orient='records', lines=True)

    def load_df_from_json(self, df_path: str, dist_path: str):
        """
        Load arxiv df and distribution dict from json file.
        Load infos to object variables.

        Parameters
        ----------
        df_path: str
            path to dataframe json
        dist_path: str
            path to distrobution json

        Returns
        -------

        """
        self.arxiv_df = pd.read_json(df_path, lines=True)

        with open(dist_path, 'r') as jsonfile:
            info = json.load(jsonfile)

        self.arxiv_distribution_reduced = info['reduced']
        self.arxiv_distribution = info['all']

    def reduce_arxiv_dataset_by_distribution(self, threshold_instances: int, research_field_num: int) -> int:
        """
        Calculate Distribution of Research Fields in whole dataset of arxiv papers.
        Save reduced dataset that has the same label distribution regarding the selected
        most frequent Reseach fields.

        Parameters
        ----------
        threshold_instances : int
            theshold for number of papers that are used as instances
        research_field_num :
            number of most frequent researchfields that get sampled

        Returns
        -------
        """
        self.arxiv_distribution = self._get_arxiv_distribution()
        relevant_labels, irelevant_labels, research_field_num = self._get_frequent_labels(research_field_num,
                                                                                          self.arxiv_distribution)
        self._drop_rare_labels(irelevant_labels)

        # sum of all research fields. One paper with 3 Research Fields counts as 3
        df_length = sum([value for key, value in self.arxiv_distribution.items() if key in relevant_labels])

        # dict with absolute numbers of apperance in reduced dataset in relation to threshold
        self.arxiv_distribution_reduced = {label: int(count / df_length * threshold_instances) for label, count in
                                           self.arxiv_distribution.items() if label in relevant_labels}

        info_path = os.path.join(FILE_PATH, "info-arxiv-distribution_" + ".json")
        meta_dist = {
            'all': self.arxiv_distribution,
            'reduced': self.arxiv_distribution_reduced
        }
        self._save_dict(info_path, meta_dist)
        self._select_labels(relevant_labels, self.arxiv_distribution_reduced)

        return research_field_num

    def add_new_data(
            self,
            relative_path: str,
            new_research_field_num: int,
            experiment_path: str,
            new_label_limit: int) -> int:
        """
        Adds new data to existing dataframe
        New data needs to have csv structure and contain relevant labels.
        ['title', 'abstract', 'categories', 'doi']

        Parameters
        ----------
        relative_path : str
            relative path from this file to dataframe
        new_research_field_num : str
            number of researchfield that should be maximally added
        experiment_path: str
            absolute path to experiment folder
        new_label_limit : int
            limit of papers that get added to dataset per label
        Returns
        -------
        int:
            number of different arxiv labels
        """
        new_df = self._load_new_data(relative_path)

        new_df = self._prevent_data_duplication(new_df)
        data_distribution_merge = self._get_merge_distribution(new_df)
        relevant_labels, irelevant_labels, _ = self._get_frequent_labels(new_research_field_num,
                                                                         data_distribution_merge)

        arxiv_reduced_labels = [self.mapping_arxiv_orkg[label] for label in self.arxiv_distribution_reduced.keys()]
        new_labels = [label for label in relevant_labels if label not in arxiv_reduced_labels]
        new_nolabels = [label for label in irelevant_labels if label not in arxiv_reduced_labels]

        merge_df, all_distribution = self._create_new_df(new_df, new_labels, data_distribution_merge, new_label_limit)
        nolabel_df, self.arxiv_distribution_reduced = self._add_nolabel_category(new_df, all_distribution, new_nolabels)

        info_path = os.path.join(experiment_path, "info-arxiv-distribution_all.json")
        self._save_dict(info_path, self.arxiv_distribution_reduced)
        self._merge_dataframes(merge_df, nolabel_df)

        return len(all_distribution)

    def unlabel_categories(self, categories: List[str], experiment_path: str):
        """
        Change name of unwanted labels to nolabel.
        Labels will be assigned to category nolabel.

        Parameters
        ----------
        experiment_path : str
            Path to experiment folder
        categories : List[str]
            List of unwanted labels

        Returns
        -------
        """
        for category in categories:
            label_num = len(self.arxiv_df[self.arxiv_df['categories'] == category])
            self.arxiv_df.loc[self.arxiv_df['categories'] == category, 'categories'] = 'nolabel'
            self.arxiv_df['categories'] = self.arxiv_df['categories'].apply(_drop_label_from_string,
                                                                            search_label=category)

            self.arxiv_distribution_reduced['nolabel'] += label_num
            del self.arxiv_distribution_reduced[category]

        print(len(self.arxiv_df[self.arxiv_df['categories'] == '']))
        self.arxiv_df['categories'].replace('', np.nan, inplace=True)
        self.arxiv_df.dropna(subset=['categories'], inplace=True)

        info_path = os.path.join(experiment_path, "info-arxiv-distribution_all.json")
        self._save_dict(info_path, self.arxiv_distribution_reduced)

    @staticmethod
    def _load_arxiv_orkg_mapping() -> Dict[str, str]:
        """ load mapping dict with label mapping between arxiv and orkg labels """
        json_path = os.path.join(FILE_PATH, 'mapping_arxiv_orkg.json')
        with open(json_path, 'r') as jsonfile:
            mapping = json.load(jsonfile)
        return mapping

    @staticmethod
    def _save_dict(path: str, info_dict: Dict):
        """ save metadata about distributions """
        with open(path, "w") as fp:
            json.dump(info_dict, fp)

    def _load_arxiv_data(self) -> pd.DataFrame:
        """ Load arxiv data from json to dataframe and drop irelevant columns"""
        if not os.path.exists(os.path.join(FILE_PATH, 'arxiv-metadata-shuffled.json')):
            shuffle_data()

        df = pd.read_json(os.path.join(FILE_PATH, 'arxiv-metadata-shuffled.json'), lines=True)
        df['categories'] = df['categories'].apply(_map_legacy)

        for column in list(df.columns):
            if column not in self.relevant_columns:
                df.drop(columns=column, inplace=True)
        return df

    def _load_new_data(self, relative_path):
        """ Load new data from csv to dataframe, rename columns and drop irelevant columns and nans"""
        df = pd.read_csv(os.path.join(FILE_PATH, relative_path))
        df = df.dropna(subset=['title', 'abstract'])
        df.categories = df.categories.str.replace('Information science', 'Information Science', regex=False)

        if 'label' in df:
            df.rename(columns={'label': 'categories'}, inplace=True)

        for column in list(df.columns):
            if column not in self.relevant_columns:
                df.drop(columns=column, inplace=True)

        return df

    def _get_arxiv_distribution(self):
        """ get distribution of arxiv dataset labels """
        distribution_dict = {}
        for label in self.arxiv_labels:
            self.arxiv_df['label_truth'] = self.arxiv_df['categories'].apply(_label_check, search_label=label)
            try:
                distribution_dict[label] = int(self.arxiv_df['label_truth'].value_counts()[1])
            except KeyError:
                pass

        self.arxiv_df = self.arxiv_df.drop('label_truth', 1)
        return distribution_dict

    def _get_merge_distribution(self, df):
        """ get distribution of new dataset labels """
        merge_distribution_dict = {}
        for label in df['categories'].unique():
            df['label_truth'] = df['categories'].apply(lambda x: label == x)
            try:
                merge_distribution_dict[label] = int(df['label_truth'].value_counts()[1])
            except KeyError:
                pass

        if 'Science' in merge_distribution_dict:
            del merge_distribution_dict['Science']

        return merge_distribution_dict

    def _get_frequent_labels(self, research_field_num: int, distribution_dict: Dict) -> Tuple[List, List, int]:
        """ sorts labels and returns the most frequent ones"""
        sorted_dist = {k: v for k, v in sorted(distribution_dict.items(), key=lambda item: item[1], reverse=True)}
        research_field_num = min(len(sorted_dist), research_field_num)

        relevant_labels = list(sorted_dist.keys())[:research_field_num]
        irelevant_labels = list(sorted_dist.keys())[research_field_num:]

        return relevant_labels, irelevant_labels, research_field_num

    def _drop_rare_labels(self, irelevant_labels):
        """ drop rare labels from dataframe"""
        for label in irelevant_labels:
            self.arxiv_df = self.arxiv_df[~self.arxiv_df['categories'].apply(_label_check, search_label=label)]
            self.arxiv_df.reset_index(inplace=True, drop=True)

    def _prevent_data_duplication(self, df) -> pd.DataFrame:
        """ First search similar dois in orkg data. Drop instance if it is a duplicat."""
        duplicate_indexes = [index for index, doi in enumerate(df.doi.to_list()) if doi in self.arxiv_df.doi.values]
        df.drop(df.index[duplicate_indexes], inplace=True)
        return df

    def _select_labels(self, relevant_labels, distribution_reduced):
        """ select subset of labels while remaining the distribution of them """
        selected_labels = []

        for label in relevant_labels:
            index_list = self.arxiv_df[self.arxiv_df['categories'].apply(_label_check, search_label=label)].index
            index_samples = random.sample(list(index_list), distribution_reduced[label])

            for index in index_samples:

                # replace doubled instances
                while index in selected_labels:
                    index = random.sample(list(index_list), 1)[0]
                selected_labels.append(index)

        # select instances via index
        self.arxiv_df = self.arxiv_df.loc[self.arxiv_df.index[selected_labels]]

    def _add_nolabel_category(
            self,
            new_df: pd.DataFrame,
            distribution_dict: Dict,
            irelevant_labels: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """ add nolabel as category for labels that arent frequent enough in new data """
        selected_labels = []

        for label in irelevant_labels:
            if label == 'Science':
                continue

            label_df = new_df[new_df.categories == label]
            selected_labels += list(label_df.index)

        distribution_dict['nolabel'] = len(selected_labels)
        df = new_df.iloc[selected_labels]
        df = df.assign(categories='nolabel')

        return df, distribution_dict

    def _create_new_df(
            self,
            merge_df: pd.DataFrame,
            new_labels: List[str],
            distribution_dict,
            sample_limit: int) -> Tuple[pd.DataFrame, Dict]:
        """ creates new dataframe from relevant labels in new data """
        all_data_dist = self.arxiv_distribution_reduced
        merge_df.reset_index(inplace=True, drop=True)
        index_list = []

        for label in new_labels:
            label_df = merge_df[merge_df.categories == label]
            index_list += list(label_df.sample(min(distribution_dict[label], sample_limit)).index)
            all_data_dist[label] = min(distribution_dict[label], sample_limit)

        df = merge_df.iloc[index_list]
        df.categories = df.categories.str.replace(' ', '')

        return df, all_data_dist

    def _merge_dataframes(self, df1, df2):
        """ concatenates 3 dataframes """
        if 'label_truth' in df1:
            df1.drop('label_truth', axis=1, inplace=True)
        if 'label_truth' in df2:
            df2.drop('label_truth', axis=1, inplace=True)
        if 'label_truth' in self.arxiv_df:
            self.arxiv_df.drop('label_truth', axis=1, inplace=True)

        self.arxiv_df = pd.concat([self.arxiv_df, df1, df2], ignore_index=True)


def shuffle_data():
    """
    Read arxiv data in pandas df and shuffle the data.
    Save shuffled data as json records.

    Returns
    -------
    """
    arxiv_hint = 'Download arxiv data from https://www.kaggle.com/Cornell-University/arxiv'
    assert os.path.exists(os.path.join(FILE_PATH, 'arxiv-metadata-oai-snapshot.json')), arxiv_hint

    file_path = os.path.join(FILE_PATH, 'arxiv-metadata-oai-snapshot.json')
    # file_path = os.path.join(FILE_PATH, 'arxiv_data/arxiv-metadata-reduced.json')

    # converting json dataset from dictionary to dataframe
    df = pd.read_json(file_path, lines=True)

    df = shuffle(df)
    df.reset_index(inplace=True, drop=True)

    # save to json file
    shuffle_path = os.path.join(FILE_PATH, 'arxiv-metadata-shuffled.json')
    df.to_json(shuffle_path, orient='records', lines=True)


legacy_mapping_path = os.path.join(FILE_PATH, 'mapping_arxiv_legacy.json')
with open(legacy_mapping_path, 'r') as infile:
    legacy_mapping = json.load(infile)


def _map_legacy(label_string: str):
    """ map old labels to new ones """
    label_list = []
    for label in label_string.split(' '):
        label_list.append(legacy_mapping.get(label, label))

    return ' '.join(label_list)


def _label_check(label_string: str, search_label: str):
    """ Mask labels with Booleans that appear in row """
    for label in label_string.split(' '):
        if search_label == label:
            return True

    return False


def _drop_label_from_string(label_string: str, search_label: str):
    """ Mask labels with Booleans that appear in row """
    valid_labels = []
    for label in label_string.split(' '):
        if search_label == label:
            continue
        valid_labels.append(label)

    return ' '.join(valid_labels)
