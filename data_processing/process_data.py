import concurrent.futures
import json
import os
import re
import time
from typing import Dict

from fuzzywuzzy import process
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from data_processing.api_data.api_data import APIData
from data_processing.orkg_data.Strategy import Strategy
from data_processing.util import recursive_items, process_abstract_string
from logs.my_logger import MyLogger

logger = MyLogger('label_data').logger
FILE_PATH = os.path.dirname(__file__)


class ORKGData:
    """
    Provides functionality to
        - load meta data for papers from orkg
        - query missing data from crossref and semnatic scholar api
        - map research fields from crossref and semantic schoolar to orkg research fields
        - collect and visualize data statistics for the orkg
    """

    def __init__(self, strategy: Strategy) -> None:
        """
        Load data from ORKG API or rdfDump

        Parameters
        ----------
        strategy :
            Strategy to load Data
        """
        self._strategy = strategy
        self.scheduler = []
        self.paper_index_list = []

        self.predicate_id = 'http://orkg.org/orkg/predicate/P30'
        self.meta_ids = {
            'doi': 'http://orkg.org/orkg/predicate/P26',
            'author': 'http://orkg.org/orkg/predicate/P27',
            'publication month': 'http://orkg.org/orkg/predicate/P28',
            'publication year': 'http://orkg.org/orkg/predicate/P29',
            'title': 'http://www.w3.org/2000/01/rdf-schema#label',
            'publisher': 'http://orkg.org/orkg/predicate/HAS_VENUE',
            'url': 'http://orkg.org/orkg/predicate/url'
        }

        self.df = pd.DataFrame(columns=['abstract', 'author', 'doi', 'url', 'publication month', 'publication year',
                                        'title', 'paper_id', 'publisher', 'crossref_field', 'semantic_field', 'label'])

        self.data_stats = {'num_papers': 0, 'num_dois': [], 'num_publisher': [], 'num_science_labels': [],
                           'num_urls': []}

    @property
    def strategy(self) -> Strategy:
        """Load Strategy for ORKG Data"""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def load_label_data(self) -> None:
        """
        Initializes dataframe with orkg data.

        Returns
        -------
        """
        predicate_statements = self._strategy.get_statement_by_predicate(self.predicate_id)
        self.df['label'] = predicate_statements['label']
        self.df['paper_id'] = predicate_statements['paper']

        subject_statements = self._strategy.get_statement_by_subject(predicate_statements['paper'], self.meta_ids)
        for column_name, values in subject_statements.items():
            if column_name == 'abstract' or column_name == 'title':
                values = [process_abstract_string(value) for value in values]
            self.df[column_name] = values

        if 'paper_id' in self.df:
            del self.df['paper_id']

    def update_dataframe(self, dataframe_path):
        """
        Load already queried information from saved datframe to self.df

        Parameters
        ----------
        dataframe_path :
            path to csv file that holds paper info

        Returns
        -------
        """
        extra_data = self.load_df_from_csv(dataframe_path)
        for index in self.df.index:
            index_doi = self.df.loc[index, 'doi'] if self.df.loc[index, 'doi'] else None
            row = extra_data[extra_data.doi == index_doi]
            if row.empty:
                row = extra_data[extra_data.title == self.df.loc[index, 'title']]

            if not row.empty:
                self.df.loc[index] = extra_data.loc[row.index].values.tolist()[0]
            else:
                self.paper_index_list.append(index)

    def add_data_from_apis(self) -> None:
        """
        Queries only papers with missing abstract or doi. \n
        First search in Crossref api for missing data.
        If there is no abstract in crossref_api or no doi exists for the paper query semantic scholar
        Use multi threading.

        Returns
        -------
        """
        if not self.paper_index_list:
            self.paper_index_list = list(range(len(self.df)))

        api_data = APIData(self.df)
        with tqdm(total=len(self.paper_index_list)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                for index in self.paper_index_list:
                    doi = self.df.at[index, 'doi']
                    if len(self.df.at[index, 'abstract']) > 50 and doi:
                        pbar.update(1)
                        continue

                    executor.submit(self._call_apis, api_data, doi, index, pbar)

    def evaluate_labels(self, column_name: str = 'crossref_field') -> Dict[str, str]:
        """
        Creates a mapping between orkg research fields and research fields from semantic scholar and crossref
        Policy:\n
        1. string matching\n
        2. fuzzy string matching:
            - value > 95 -> best match
            - 90 < value < 95 -> longest match

        Parameters
        ----------
        column_name : str
            api name for mapping

        Returns
        -------
        Dict[str, str]
        """
        research_field_mapping = {}

        with open(os.path.join(FILE_PATH, 'data/data_ResearchFields.json'), 'r') as infile:
            research_fields = json.load(infile)

        # get all names of Research Fields from dict
        research_fields_values = [value for value in recursive_items(research_fields)]

        science_df = self.df.query('label == "Science"')

        # replace Science label if possible
        for index, labels in zip(list(science_df.index.values), science_df[column_name]):
            if type(labels) is str:

                for label in labels.split(','):
                    label = re.sub('[^a-zA-Z ]+', '', label)
                    label = label.strip()

                    if label in research_fields_values:
                        logger.info(label)
                        science_df.at[index, 'label'] = label
                        break
                    elif len(label) > 3:
                        if label in research_field_mapping.keys():
                            continue

                        # fuzzy string matching for unknown labels
                        matching_fields = process.extract(label, research_fields_values, limit=5)
                        best_matching_fields = {label: value for label, value in matching_fields if value >= 90}

                        if best_matching_fields:
                            highest_score_label = str(max(best_matching_fields, key=best_matching_fields.get))
                            if best_matching_fields[highest_score_label] >= 95:
                                best_label = highest_score_label
                            else:
                                # longest match is preferred
                                best_label = str(max(best_matching_fields.keys(), key=len))
                            logger.info(f'{label} -> {best_label}')
                            research_field_mapping[label] = best_label

        return research_field_mapping

    def map_labels(self, column_name='crossref_field'):
        """
        maps Science Research field and overwrites old Research Field in df['label']

        Parameters
        ----------
        column_name :
            api name for mapping research fields

        Returns
        -------
        """
        path = os.path.join(FILE_PATH, 'data/research_field_mapping_' + column_name + '.json')
        with open(path, 'r') as infile:
            mapping = json.load(infile)

        science_df = self.df.query('label == "Science"')

        for index, labels in zip(list(science_df.index.values), science_df[column_name]):
            if type(labels) is str:

                for label in labels.split(','):
                    label = re.sub('[^a-zA-Z ]+', '', label)
                    label = label.strip()

                    if label in mapping.keys():
                        self.df.at[index, 'label'] = mapping[label]
                        break

    @staticmethod
    def save_field_mapping(field_map_dict: Dict, column_name='crossref_field') -> None:
        """
        Parameters
        ----------
        field_map_dict : Dict
            mapping dictionary
        column_name :
            api name for mapping research fields

        Returns
        -------
        """
        path = os.path.join(FILE_PATH, 'data/research_field_mapping_' + column_name + '.json')
        with open(path, 'w') as fp:
            json.dump(field_map_dict, fp)

    def load_df_from_csv(self, path: str) -> pd.DataFrame:
        """
        Parameters
        ----------
        path :
            absolute path to csv file

        Returns
        -------
        pd.Dataframe:
        """
        df = pd.read_csv(path)
        df['publication month'] = df['publication month'].fillna(0).astype(int, errors='ignore')
        df['publication year'] = df['publication year'].fillna(0).astype(int)
        df = df.fillna('')
        df['abstract'] = df['abstract'].astype(str).apply(process_abstract_string)

        return df

    def add_data_statistics(self) -> None:
        """ ads statistics of meta data completness """
        self.data_stats['num_papers'] = self.df.shape[0]
        self.data_stats['num_dois'].append(self.df.query('doi == ""').shape[0])
        self.data_stats['num_science_labels'].append(self.df.query('label == "Science"').shape[0])
        self.data_stats['num_publisher'].append(self.df.query('publisher == ""').shape[0])
        self.data_stats['num_urls'].append(self.df.query('url == ""').shape[0])

    def plot_data_statistics(self) -> None:
        """
        Plots statistics for collected meta data .
        Expects added data for 'RDF Dump', 'RDF Dump + api_data', 'crossref', 'semantic_scholar'.
        """
        sns.set()
        keys = ['ORKG API', 'ORKG API + api_data', 'crossref', 'semantic_scholar']

        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Missing Meta-Data Statistics for Papers in ORKG')

        sns.barplot(ax=axes[0, 0], x=keys[:2], y=self.data_stats['num_publisher'][:2])
        axes[0, 0].set_title("No Published In")
        axes[0, 0].axhline(self.data_stats['num_papers'], ls='--')

        sns.barplot(ax=axes[1, 0], x=keys[:2], y=self.data_stats['num_dois'][:2])
        axes[1, 0].set_title("No DOI")
        axes[1, 0].axhline(self.data_stats['num_papers'], ls='--')

        sns.barplot(ax=axes[0, 1], x=keys[:2], y=self.data_stats['num_urls'][:2])
        axes[0, 1].set_title("No URL")
        axes[0, 1].axhline(self.data_stats['num_papers'], ls='--')

        sns.barplot(ax=axes[1, 1], x=keys[1:], y=self.data_stats['num_science_labels'][1:])
        axes[1, 1].set_title('Science as Research Field')
        axes[1, 1].axhline(self.data_stats['num_papers'], ls='--')

        plt.show()

    def _call_apis(self, api_data: APIData, doi: str, index: int, pbar):
        """ call apis with multiple threads """
        try:
            data = api_data.get_crossref_data(doi, index)
            if 'abstract' not in data or 'doi' not in data:
                data = api_data.get_semantic_scholar_data(doi, index)

            self._update_dataframe_with_api_data(data, index)
            pbar.update(1)

        except ConnectionError:
            time.sleep(60)
            logger.error('Connection Error')

    def _update_dataframe_with_api_data(self, data: Dict, index: int):
        """ write data from dict to dataframe , extra check for publisher and url"""
        self.df.at[index, 'abstract'] = data.get('abstract', self.df.at[index, 'abstract'])
        self.df.at[index, 'crossref_field'] = data.get('crossref_field', self.df.at[index, 'crossref_field'])
        self.df.at[index, 'semantic_field'] = data.get('semantic_field', self.df.at[index, 'semantic_field'])

        if not self.df.at[index, 'publisher']:
            self.df.at[index, 'publisher'] = data.get('publisher', '')

        if not self.df.at[index, 'doi']:
            self.df.at[index, 'doi'] = data.get('doi', '')

        if not self.df.at[index, 'url']:
            self.df.at[index, 'url'] = data.get('url', '')
