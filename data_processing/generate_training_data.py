import math
import pandas as pd
import numpy as np
import json
from typing import List, Dict
import random
import os
from data_processing.util import recursive_items, delete_files, process_abstract_string

FILE_PATH = os.path.dirname(__file__)
pd.set_option("display.max_columns", None)


class GenerateData:
    """
    Provides functions to prepare Training Data from pandas df
    """

    def __init__(self, df: pd.DataFrame = pd.DataFrame()):
        """
        initalizes object with optional dataframe
        df needs title, author, publisher and url all as string

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe of data that needs to be prepared for trianing
        """
        self.df = df
        self.research_fields = []
        self.all_research_fields = []
        self.test_df = pd.DataFrame(columns=['abstract', 'author', 'title', 'publisher', 'label'])
        self.train_df = pd.DataFrame(columns=['abstract', 'author', 'title', 'publisher', 'label'])

    def load_df_from_csv(self, path: str) -> None:
        """
        Loads dataframe from csv.
        Only papers without the Science label will be used.

        Parameters
        ----------
        path:
            relativ path from script folder to csv folder

        Returns
        -------
        """
        self.df = pd.read_csv(os.path.join(FILE_PATH, path))
        self.df = self.df.fillna('')
        self.df = self.df.query('label != "Science"')

    def load_all_research_fields(self) -> None:
        """
        Loads all research fields that exist in the orkg.

        Returns
        -------
        """
        json_path = os.path.join(FILE_PATH, 'data/data_ResearchFields.json')

        with open(json_path, 'r') as infile:
            self.research_fields = json.load(infile)
        self.all_research_fields = [value for value in recursive_items(self.research_fields)]

    def load_available_research_fields(self) -> None:
        """
        Loads only research fields that exist in the dataframe.

        Returns
        -------
        """
        self.all_research_fields = self.df.label.unique().tolist()

    def select_relevant_df_columns(self, columns: List[str]) -> None:
        """
        Drops unneccesary columns from dataframe

        Parameters
        ----------
        columns: List[str]
            List of column names

        Returns
        -------
        """
        self.df = self.df[columns]
        self.df.replace('', np.nan, inplace=True)
        self.df.dropna(inplace=True)

    def select_relevant_research_fields(self, threshold: int = 1, max_num: int = 1000) -> None:
        """
        Filter labels in dataframe based on the number of appearance and a maximum value for amount of different labels

        Parameters
        ----------
        threshold : int
            min value for appearance, labels below the threshold will be dropped
        max_num : int
            maximum of labels that get accepted in dataset

        Returns
        -------
        """
        relevant_fields = []
        count = 0
        label_set = self.df.label.value_counts(ascending=True)

        # find all labels above threshold until max_num is reached
        for num_values, label in zip(label_set.values.tolist(), label_set.index.tolist()):

            if num_values > threshold:
                relevant_fields.append(label)
                count += 1

            if count > max_num:
                break

        # filter df for labels in list
        self.df = self.df[self.df['label'].isin(relevant_fields)]

    def split_data_set(self, split_ratio: float) -> None:
        """
        Saves to dataframes of train and test data.
        Generates train and test regarding the split counts
        If only 1 instance is available it gets added to train set

        Parameters
        ----------
        split_ratio: float
            percentage of the training set

        Returns
        -------
        """
        label_set = self.df.label.value_counts(ascending=True)

        split_counts = self._get_split_counts(label_set, split_ratio)

        for count, label in zip(label_set.values.tolist(), label_set.index.tolist()):
            query_string = 'label == "' + label + '"'

            if count == 1:
                # if only 1 instance in dataset just add to train data
                self.train_df = self.train_df.append(self.df.query(query_string))
            else:
                # make split for every research field
                split = split_counts[label]

                field_df = self.df.query(query_string)
                field_df = field_df.sample(frac=1).reset_index(drop=True)

                for i in range(split):
                    self.train_df = self.train_df.append(field_df.iloc[[i]])

                for j in range(split, count):
                    self.test_df = self.test_df.append(field_df.iloc[[j]])

        self.train_df = self.train_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)

    def _get_split_counts(self, label_set: pd.Series, split_ratio: float) -> Dict:
        """
        Creates split for every research field seperately.
        Corrects overall split with research fields that appear frequently (over threshold).

        Parameters
        ----------
        label_set: pd.Series
            Series with numbers of appearance for each Research Field
        split_ratio: float
            percentage of the training set

        Returns
        -------
        Dict
        """
        threshold = 50
        frequent_fields = []
        split_counts = {}
        total_count = 0

        total_split = int(self.df.shape[0] * split_ratio)

        # noinspection PyTypeChecker
        for count, label in zip(label_set.values.tolist(), label_set.index.tolist()):

            split = math.floor(count * split_ratio) if random.random() >= 0.5 else math.floor(count * split_ratio)
            # just split a small percentage of rare data to test set
            if count <= 5 and random.random() < 0.9:
                split = count

            split_counts[label] = split
            total_count += split

            if count > threshold:
                frequent_fields.append(label)

        # correct split
        while total_split > total_count:
            split_counts[random.choice(frequent_fields)] += 1
            total_count += 1

        while total_split < total_count:
            split_counts[random.choice(frequent_fields)] -= 1
            total_count -= 1

        return split_counts

    def generate_instances(self,
                           filename: str,
                           columns: List[str],
                           false_percentage: float,
                           create_false_data: bool = False
                           ) -> None:
        """
        Generates Training and tests instances. Shuffles the instances and writes them to .txt file
        in classifier/training_data folder. \n
        Instances have the following structure:
        bool	[test].txt	  [Research Field]	  description

        Parameters
        ----------
        filename:
            end of filename
        columns:
            columns that provide description data
        false_percentage:
            ratio of false instances for training data
        create_false_data: bool
            Optional bool that enables creation of false instances

        Returns
        -------
        """
        train_instances = []
        test_instances = []

        train_instances += self._generate_true_instances(self.train_df, 'test', columns)
        if create_false_data:
            train_instances += self._generate_false_train_instances(self.train_df, 'test', columns,
                                                                    percentage=false_percentage)
        random.shuffle(train_instances)

        test_instances += (self._generate_true_instances(self.test_df, 'test', columns))
        if create_false_data:
            test_instances += (self._generate_false_test_instances(self.test_df, 'test', columns))
        random.shuffle(test_instances)

        self.write_data_to_file('train_' + filename, train_instances)
        self.write_data_to_file('test_' + filename, test_instances)

    def _generate_true_instances(self, df: pd.DataFrame, filename: str, columns: List[str]) -> List[str]:
        """
        Creates true instances for all rows in dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            dataframe with instances
        filename: str
            test or train
        columns: List[str]
            names of relevant columns for description

        Returns
        -------
        List[str]
        """

        file = '[' + filename + '].txt'
        instances = []

        for field in self.all_research_fields:

            query_string = 'label == "' + field + '"'
            field_df = df.query(query_string)

            if not field_df.empty:
                for index in (list(field_df.index.values)):
                    research_field = field_df.at[index, 'label']
                    description = GenerateData._create_description(field_df, index, columns)

                    instance = 'true' + '\t' + file + '\t' + '[' + research_field + ']' + '\t' + description + '\n'
                    instances.append(instance)

        return instances

    def _generate_false_train_instances(self,
                                        df: pd.DataFrame,
                                        filename: str,
                                        columns: List,
                                        percentage=0.4) -> List[str]:
        """
        Creates a percentage of random false training instances.

        Parameters
        ----------
        df: pd.DataFrame
            dataframe with instances
        filename: str
            test or train
        columns: List[str]
            names of relevant columns for description
        percentage
            percentage of false instances to create

        Returns
        -------

        """
        file = '[' + filename + '].txt'
        instances = []
        research_fields_prio_list = df.label.unique().tolist()

        frequent_fields = [key for key, count in self.df.label.value_counts().to_dict().items() if count > 20]

        num_instances = int(len(df.index.tolist()) * percentage)

        for _ in range(num_instances):
            # first select one entry of each research field after that pick random fields
            if research_fields_prio_list:
                prio_field = research_fields_prio_list.pop()
                prio_df = df.query('label == "' + prio_field + '"')
                index = random.choice(list(prio_df.index.values))
            else:
                # just choose frequent fields randomly
                index = random.choice(list(df.index.values))
                while df.at[index, 'label'] not in frequent_fields:
                    index = random.choice(list(df.index.values))

            instance_field = df.at[index, 'label']

            random_field = random.choice(self.all_research_fields)

            while random_field == instance_field:
                random_field = random.choice(self.all_research_fields)

            description = GenerateData._create_description(df, index, columns)
            instance = 'false' + '\t' + file + '\t' + '[' + random_field + ']' + '\t' + description + '\n'
            instances.append(instance)

        return instances

    def _generate_false_test_instances(self, df: pd.DataFrame, filename: str, columns: List) -> List[str]:
        """
        Creates all false instances for every true instance.

        Parameters
        ----------
        df: pd.DataFrame
            dataframe with instances
        filename: str
            test or train
        columns: List[str]
            names of relevant columns for description

        Returns
        -------
        List[str]
        """
        file = '[' + filename + '].txt'
        instances = []
        num_test_instances = self.test_df.shape[0]

        for index in range(num_test_instances):

            for false_field in self.all_research_fields:

                instance_field = df.at[index, 'label']
                if false_field == instance_field:
                    continue

                description = GenerateData._create_description(df, index, columns)
                instance = 'false' + '\t' + file + '\t' + '[' + false_field + ']' + '\t' + description + '\n'
                instances.append(instance)

        return instances

    @staticmethod
    def _create_description(df, index: int, columns: List) -> str:
        """
        Creates the description for an intance.
        Preprocess the string.

        Parameters
        ----------
        df : pd.DataFrame
            dataframe with instances
        index : int
            index of instance in dataframe
        columns : List
            names of relevant columns for description text

        Returns
        -------
        str
        """
        description = ".".join([df.at[index, col_name] for col_name in columns])
        encoded_string = description.encode("ascii", "ignore")
        description = encoded_string.decode()
        description = description.replace('\n', '')
        description = description.replace('\t', '')
        description = description.replace('\r', '')
        description = description.replace('[', '')
        description = description.replace(']', '')

        if 'abstract' in columns:
            description = process_abstract_string(description)

        return description

    @staticmethod
    def write_data_to_file(experiment_name: str, instances: List[str]) -> None:
        """
        Writes datato file.

        Parameters
        ----------
        experiment_name: str
            experiment_name
        instances: List[str]
            all instances that belong to experiment

        Returns
        -------
        """
        # delete old files
        with open(os.path.join(FILE_PATH, '../classifier/training_data/' + experiment_name + '.txt'), 'a') as testfile:
            for instance in instances:
                testfile.write(instance)

    def generate_training_data(self,
                               threshold_fields: int,
                               max_num_fields: int,
                               data_split: float,
                               experiment_names: List[str],
                               percentage_false_training_data: float
                               ) -> None:
        """
        Take input arguments and create the Training data based on that configs

        Parameters
        ----------
        threshold_fields : int
        max_num_fields : int
        data_split: float
        experiment_names : List[str]
        percentage_false_training_data : float

        Returns
        -------
        """
        delete_files(os.path.join(FILE_PATH, '../classifier/training_data/'), '.txt')

        self.load_df_from_csv('data/clean_data_all.csv')
        self.select_relevant_df_columns(['abstract', 'author', 'title', 'publisher', 'label'])
        self.select_relevant_research_fields(threshold=threshold_fields, max_num=max_num_fields)
        self.load_available_research_fields()

        self.split_data_set(data_split)

        for exp_name in experiment_names:
            self.generate_instances(exp_name, exp_name.split('_'), percentage_false_training_data)


if __name__ == '__main__':
    generate = GenerateData()
    delete_files(os.path.join(FILE_PATH, '../classifier/training_data/'), '.txt')

    generate.load_df_from_csv('data/clean_data_all.csv')
    generate.select_relevant_df_columns(['abstract', 'author', 'title', 'publisher', 'label'])
    generate.select_relevant_research_fields(threshold=20, max_num=1000)
    generate.load_available_research_fields()

    generate.split_data_set(0.7)

    file_name = 'title'
    column = ['title']
    percentage_false_instances = 0.4
    generate.generate_instances(file_name, column, percentage_false_instances)

    file_name = 'title_publisher'
    column = ['title', 'publisher']
    generate.generate_instances(file_name, column, percentage_false_instances)

    file_name = 'title_abstract'
    column = ['title', 'abstract']
    generate.generate_instances(file_name, column, percentage_false_instances)
