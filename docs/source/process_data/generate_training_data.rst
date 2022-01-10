
======================
Generate Training Data
======================

.. role:: python(code)
    :language: python

Generate Training Data from Dataframe Data.
This class provides data for BERT classification.

.. code-block:: python

    # init class and remove old training data with extension .txt
    generate = GenerateData()
    delete_files(os.path.join(FILE_PATH, '../classifier/training_data/'), '.txt')

    # load data from dataframe and reduce df to relevant columns
    generate.load_df_from_csv('data/clean_data_all.csv')
    generate.select_relevant_df_columns(['abstract', 'author', 'title', 'publisher', 'label'])
    generate.select_relevant_research_fields(threshold=100, max_num=10)
    generate.load_available_research_fields()

.. code-block:: python

    # 70% train 30% test and 40% False instances in training data
    generate.split_data_set(0.7)
    percentage_false_instances = 0.4

    # setup experiment infos
    file_name = 'title'
    column = ['title']

    # create Instances and save them to file: '../classifier/training_data/'
    generate.generate_instances(file_name, column, percentage_false_instances)

.. automodule:: data_processing.generate_training_data
    :members:
