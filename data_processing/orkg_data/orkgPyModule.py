from typing import List, Dict
from data_processing.orkg_data.Strategy import Strategy
from orkg import ORKG
from requests.exceptions import ConnectionError
import time
import requests
import json


class ORKGPyModule(Strategy):
    """
    Gets meta data ORKG for papers from ORKG API
    """

    def __init__(self):
        self.orkg = ORKG(host="http://orkg.org/orkg/")
        self.predicate_url = 'http://www.orkg.org/orkg/api/statements/predicate/'
        self.subject_url = 'http://www.orkg.org/orkg/api/statements/subject/'

    def get_statement_by_predicate(self, predicate_id: str) -> Dict[str, List]:
        """
        Provides all paper ids and titles that have a research field.

        Parameters
        ----------
        predicate_id : str
            ID of "has research field"

        Returns
        -------
        Dict[str, list]
        """
        predicate_id = predicate_id.split('/')[-1]
        statement_data = {'paper': [], 'label': []}
        size = '1000'

        for count in range(99):
            try:
                response = requests.get(self.predicate_url + predicate_id + '?size=' + size + '&page=' + str(count))
            except ConnectionError:
                time.sleep(60)
                response = requests.get(self.predicate_url + predicate_id + '?size=' + size + '&page=' + str(count))

            if response.ok:
                content = json.loads(response.content)['content']

                for statement in content:
                    statement_data['paper'].append(statement['subject']['id'])
                    statement_data['label'].append(statement['object']['label'])

                if len(content) < int(size):
                    break

        print('ready')
        return statement_data

    def get_statement_by_subject(self, paper_ids: List, meta_ids: Dict) -> Dict[str, list]:
        """
        Stores meta_infos for each paper in a Dict.
        Dict = {column_name: List[str], ...}

        Parameters
        ----------
        paper_ids : List[str]
            all paper_ids in orkg
        meta_ids : Dict
            relevant meta_ids (doi, ...)

        Returns
        -------
        Dict[str, list]
        """
        meta_infos = {key: [] for key in meta_ids.keys()}

        for key, meta_id in meta_ids.items():
            meta_ids[key] = meta_id.split('/')[-1]

        # structure: {predicate_id: predicate_string}
        look_up = {v: k for k, v in meta_ids.items()}

        for paper_id in paper_ids:

            try:
                response = requests.get(self.subject_url + paper_id + '?size=100')
            except ConnectionError:
                time.sleep(60)
                response = requests.get(self.subject_url + paper_id + '?size=100')

            if response.ok:
                content = json.loads(response.content)['content']
                infos = {key: [] for key in meta_ids.keys()}

                for statement in content:

                    pred_id = statement['predicate']['id']
                    if pred_id in meta_ids.values():
                        infos[look_up[pred_id]].append(statement['object']['label'])

                    if not infos['title']:
                        infos['title'].append(statement['subject']['label'])

                # build lists in meta info dict for every predicate field
                for key, value in infos.items():
                    if len(value) == 0:
                        value = ""

                    if len(value) == 1:
                        value = value[0]

                    meta_infos[key].append(value)

        return meta_infos
