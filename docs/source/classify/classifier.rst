=========================
Research Field Classifier
=========================

.. role:: python(code)
    :language: python

Available Embedders: Universal Sentence Encoder, Tfidf Sklearn and Tfidf Gensim. The second tfidf allows batch processing but has poor performance.
The classifier learns the ORKG Research Fields based on ORKG Training Data and mapped data from arXiv.
The SGD Model cab use Bayesian Optimization for Parameter Optimization.

Imports:

.. code-block:: python

    from classifier.sgd_classifier import MySGDClassifier, TUNE_SETTINGS, SGD_PARAMS, TUNE_GRID_PARAMS
    from classifier.classify_util import save_metrics, create_field_table
First define settings:

.. code-block:: python

    # EXPERIMENT META
    embedder_name = 'tfidf sklearn'
    experiments = ['title', 'title_abstract']
    batchsize = 10000000
    dataset_name = 'arxiv'
    automatic_optimization = False

    # EXPERIMENT PATH
    exp_time = datetime.datetime.now().strftime("%d_%m_%Y-%H_%M")
    exp_foldername = f"SGD_{embedder_name}__{exp_time}"
    experiment_path = os.path.join(FILE_PATH, 'experiments', exp_foldername)

    os.mkdir(experiment_path)
Run experiment:

.. code-block:: python

    # run experiments and store results in dict
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

Calculate Metrics and create Table:

.. code-block:: python

    # SAVE METRICS
    save_metrics(results, {'train setting': train_settings,
                           'tune settings': TUNE_SETTINGS,
                           'tune grid': TUNE_GRID_PARAMS,
                           'sgd params': SGD_PARAMS},
                 experiment_path)

    # SAVE EXPERIMENT DATA TO TABLE
    create_field_table(experiment_path)

.. automodule:: classifier.sgd_classifier
    :members: