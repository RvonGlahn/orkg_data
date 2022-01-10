=================
Process ORKG Data
=================

.. role:: python(code)
    :language: python

Init ORKGData with ORKGPyModule() and load labeled data from API:

.. code-block:: python

    orkg_data = ORKGData(ORKGPyModule())
    orkg_data.load_label_data()
    # create csv for labeled orkgdata
    create_csv(orkg_data.df, os.path.join(FILE_PATH, 'data/orkg_raw_data.csv'))
    # track statistics of pure orkg data
    orkg_data.add_data_statistics()

If you already have the orkgdata saved in a csv, you can load it instead:.
This applies for every step.

.. code-block:: python

    orkg_data = ORKGData(ORKGPyModule())
    orkg_data.load_df_from_csv('data/raw_data.csv')
    orkg_data.add_data_statistics()

If you already have data for abstract, doi, ... you can add it now.

.. code-block:: python

    orkg_data.update_dataframe(os.path.join(FILE_PATH, 'data_processing/data/clean_data_all.csv'))

Add data to dataframe from crossref and semantic scholar APIs:

.. code-block:: python

    orkg_data.add_data_from_apis()
    # save data before processing it may brake
    create_csv(labelling.df, os.path.join(FILE_PATH, 'data/api_data.csv'))
    orkg_data.add_data_statistics()

Select API name from which you want to map the Research Fields to ORKG Research Fields

.. code-block:: python

    mapping_column = 'crossref_field' # same for 'semantic_field'
    field_mapping = orkg_data.evaluate_labels(column_name=mapping_column)
    orkg_data.save_field_mapping(field_mapping, column_name=mapping_column)

Before applying the mapping I highly recommend you to validate the mapping json file!
It can be found in the 'processing_data/data' folder.

.. code-block:: python

    orkg_data.map_labels(column_name=mapping_column)
    create_csv(orkg_data.df, os.path.join(FILE_PATH, 'data/clean_data_crossref.csv'))
    orkg_data.add_data_statistics()

Plot Data Statisitcs for every df you added with add_data_statistics().
Expects added data for 'ORKG API', 'ORKG API + api_data', 'crossref', 'semantic_scholar'.
If you have different settings adjust the code in plot_data_statistics().

.. code-block:: python

    orkg_data.plot_data_statistics()


.. automodule:: data_processing.process_data
    :members:


