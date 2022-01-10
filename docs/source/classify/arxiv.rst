=====================
ArXiv DATA Processing
=====================

.. role:: python(code)
    :language: python

Availbale Embedders: Uinversal Sentence Encoder, Tfidf Sklearn and Tfidf Gensim. The second tfidf allows batch processing but has poor performance.
The classifier learns the ORKG Research Fields based on ORKG Training Data and mapped data from arXiv.
The SGD Model cab use Bayesian Optimization for Parameter Optimization.

Imports:

.. code-block:: python

    from classifier.arxiv_data.process_arxiv import ProcessArxivData
First define settings:

.. code-block:: python

    # DATA-PROCESSING META DATA
    label_num_max = 200
    label_num_new = 15
    threshold_instances = 200000
    new_label_limit = 2000
    train_data_percentage = 0.7
    calculate_data = True
    load_data_path = 'experiments/xxx'

    NOLABEL_CATEGORIES = [
    'stat.AP',
    'q-fin.MF',
    'q-fin.CP',
    'q-bio',
    'q-bio.QM',
    'physics.data-an',
    'stat.CO',
    'econ.GN',
    'q-fin.EC',
    'Engineering'
    ]

    # DATA AND EXPERIMENT PATHS
    data_path = '../../data_processing/data/clean_data_all.csv'
    exp_time = datetime.datetime.now().strftime("%d_%m_%Y-%H_%M")
    exp_foldername = f"SGD_{embedder_name}_instances-{threshold_instances}__{exp_time}"
    experiment_path = os.path.join(FILE_PATH, 'experiments', exp_foldername)

    os.mkdir(experiment_path)
Run experiment:

.. code-block:: python

    # PROCESS EXISTING AND ADD NEW DATA TO DATASET
    data = ProcessArxivData()

    _ = data.reduce_arxiv_dataset_by_distribution(threshold_instances, label_num_max)

    # e.g. add orkg data to arxiv data see function description for more info
    label_num_real = data.add_new_data(data_path, label_num_new, experiment_path, new_label_limit)

    # drop labels from NOLABEL_CATEGORIES, which may have a bad performance
    data.unlabel_categories(NOLABEL_CATEGORIES, experiment_path)
    data.save_df_as_json(experiment_path)

.. automodule:: classifier.arxiv_data.process_arxiv
    :members: