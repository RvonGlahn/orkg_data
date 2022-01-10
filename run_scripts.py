import datetime
import json
import os
import shutil
import warnings

from classifier.sgd_classifier import MySGDClassifier, TUNE_SETTINGS, SGD_PARAMS, TUNE_GRID_PARAMS
from classifier.arxiv_data.process_arxiv import ProcessArxivData
from classifier.classify_util import save_metrics, create_field_table
from data_processing.process_data import ORKGData
from data_processing.orkg_data.orkgPyModule import ORKGPyModule
from data_processing.util import create_csv

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
FILE_PATH = os.path.dirname(__file__)

NOLABEL_CATEGORIES = [
    'stat.AP',
    'stat.OT',
    'stat.CO',
    'q-fin.MF',
    'q-fin.CP',
    'q-fin.EC',
    'q-bio',
    'q-bio.QM',
    'q-bio.OT',
    'q-bio.SC',
    'q-bio.TO',
    'physics.data-an',
    'econ.GN',
    'econ.TH',
    'Engineering',
    'cs.CE'
]


def run_sgd_experiments():
    """
    Script that runs experiment for Logistic Regression with TFIDF and SGD with Contextual Sentence Encoder
    Use configuration options to generate different Training Data

    label_num_max:              Maximum of arxiv labels before the mapping
    label_num_new_max:          Maximum of new labels that get added to arxiv data
    threshold_instances:        Maximum of papers selected from arxiv set
    threshold_instances_new:    Maximum of papers selected from new set
    embedder_name:              name of used embedder 'tfidf sklearn', 'sentence embedding', 'tfidf'
    experiments:                determines which data gets used
    dataset_name:               info purpose
    batchsize:                  only relevant for batch_processing, otherwise needs to be set bigger than dataset size
    automatic_optimization:     if True sgd model gets optimized with bohb
    new_data_path:              path to new_dataset
    experiment_path:            path to experiment folder, where you can fidn all necessary models and info
    calculate_data:             if True calculate arxiv data new
    load_data_path:             if calculate_data this path is needed to copy the old data to new experiment
    train_split:                percentage of training data
    """
    # DATA-PROCESSING META DATA
    label_num_arxiv_max = 200
    label_num_new_max = 15
    threshold_instances_arxiv = 200000
    threshold_instances_new = 2000
    train_data_percentage = 0.7
    calculate_data = True
    load_data_path = 'experiments/SGD_tfidf sklearn_instances-200000__08_09_2021-11_34'

    # EXPERIMENT META
    embedder_name = 'tfidf sklearn'
    experiments = ['title', 'title_abstract']
    batchsize = 10000000
    dataset_name = 'arxiv'
    automatic_optimization = False

    # DATA AND EXPERIMENT PATHS
    new_data_path = '../../data_processing/data/clean_data_all.csv'
    exp_time = datetime.datetime.now().strftime("%d_%m_%Y-%H_%M")
    exp_foldername = f"SGD_{embedder_name}_instances-{threshold_instances_arxiv}__{exp_time}"
    experiment_path = os.path.join(FILE_PATH, 'experiments', exp_foldername)

    os.mkdir(experiment_path)

    # PROCESS AND ADD NEW DATA TO DATASET
    if calculate_data:
        data = ProcessArxivData()
        _ = data.reduce_arxiv_dataset_by_distribution(threshold_instances_arxiv, label_num_arxiv_max)
        label_num_real = data.add_new_data(new_data_path, label_num_new_max, experiment_path, threshold_instances_new)
        data.unlabel_categories(NOLABEL_CATEGORIES, experiment_path)
        data.save_df_as_json(experiment_path)
    else:
        # copy json file with dataset from old experiment
        shutil.copy2(os.path.join(load_data_path, 'arxiv-metadata-distributed_merged.json'), experiment_path)
        with open(os.path.join(load_data_path, 'meta-info.json'), 'r') as jsonfile:
            info = json.load(jsonfile)
            label_num_real = info['setting']['train setting']['arxiv label number']

    train_settings = {
        'embedder_name': embedder_name,
        'experiments': experiments,
        'batchsize': batchsize,
        'dataset_name': dataset_name,
        'automatic_optimization': automatic_optimization,
        'arxiv label number': label_num_real,
        'threshold instances': threshold_instances_arxiv,
    }

    results = {}

    classifier = MySGDClassifier(
        experiment_path=experiment_path,
        embedding_type=embedder_name,
        batch_size=batchsize,
        auto_opt=automatic_optimization,
        train_split=train_data_percentage
    )

    for experiment in experiments:
        classifier.set_experiment_name(experiment)
        results[experiment] = classifier.run_experiment(dataset_name)

    # SAVE METRICS
    save_metrics(results, {'train setting': train_settings,
                           'tune settings': TUNE_SETTINGS,
                           'tune grid': TUNE_GRID_PARAMS,
                           'sgd params': SGD_PARAMS},
                 experiment_path)
    create_field_table(experiment_path)

    # shutdown computer after execution
    # os.system("shutdown /s")


def run_process_data():
    """
    Script that runs the whole data collecting process.
        1. Load ORKG Data
        2. Query APIs
        3. Map Labels from APIs ti ORKG Labels
        4. Plot Statistics of collected data

    Returns
    -------
    """
    query_orkg_api = True
    update_orkg_data = True
    query_data_apis = True
    create_crossref_mapping = True
    create_semantic_scholar_mapping = True

    orkg_data = ORKGData(ORKGPyModule())

    # LOAD ORKG DATA
    if query_orkg_api:
        orkg_data.load_label_data()
        create_csv(orkg_data.df, os.path.join(FILE_PATH, 'data_processing/data/orkg_raw_data.csv'))
    else:
        orkg_data.df = orkg_data.load_df_from_csv(os.path.join(FILE_PATH, 'data_processing/data/orkg_raw_data.csv'))
    orkg_data.add_data_statistics()

    # UPDATE DATA WITH ALREADY COLLECTED DATA
    if update_orkg_data:
        orkg_data.update_dataframe(os.path.join(FILE_PATH, 'data_processing/data/clean_data_all.csv'))

    # QUERY API DATA
    if query_data_apis:
        orkg_data.add_data_from_apis()
        create_csv(orkg_data.df, os.path.join(FILE_PATH, 'data_processing/data/api_data.csv'))
        orkg_data.add_data_statistics()

    # CREATE MAPPING - crossref
    mapping_column = 'crossref_field'
    if create_crossref_mapping:
        field_mapping = orkg_data.evaluate_labels(column_name=mapping_column)
        orkg_data.save_field_mapping(field_mapping, column_name=mapping_column)

        print(f"Crossref mapping saved at: data_processing/data/research_field_mapping_{mapping_column}.json")
        confirmed = input("Checked label Mapping? (yes/no)")
        if confirmed != 'yes':
            exit(1)

    # APPLY MAPPING - crossref
    orkg_data.map_labels(column_name=mapping_column)
    create_csv(orkg_data.df, os.path.join(FILE_PATH, 'data_processing/data/clean_data_crossref.csv'))
    orkg_data.add_data_statistics()

    # CREATE MAPPING - semantic scholar
    mapping_column = 'semantic_field'
    if create_semantic_scholar_mapping:
        field_mapping = orkg_data.evaluate_labels(column_name=mapping_column)
        orkg_data.save_field_mapping(field_mapping, column_name=mapping_column)

        print(f"Semantic Scholar mapping saved at: data_processing/data/research_field_mapping_{mapping_column}.json")
        confirmed = input("Checked label Mapping? (yes/no)")
        if confirmed != 'yes':
            exit(1)

    # APPLY MAPPING - semantic scholar
    orkg_data.map_labels(column_name=mapping_column)
    create_csv(orkg_data.df, os.path.join(FILE_PATH, 'data_processing/data/clean_data_semantic_scholar.csv'))

    orkg_data.add_data_statistics()
    orkg_data.plot_data_statistics()
    # os.system("shutdown /s")


if __name__ == "__main__":
    # run_process_data()
    run_sgd_experiments()
