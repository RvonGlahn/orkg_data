========
API Data
========

.. role:: python(code)
    :language: python

DOIFinder uses Selenium for webscraping.

.. code-block:: python

    # init headless browser
    finder = DoiFinder()

    # scrape data for explicit title
    web_data = finder.scrape_data_from_semantic_scholar('paper title placeholder')

    # close headless browser session
    finder.close_session()

.. automodule:: data_processing.api_data.doi_finder
    :members:

.. automodule:: data_processing.api_data.api_data
    :members: