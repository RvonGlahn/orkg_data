============================
How to Install ResearchField
============================

.. role:: bash(code)
    :language: bash

Install from gitlab repository:

.. code-block:: bash

    git clone https://gitlab.com/TIBHannover/orkg/orkg-research-fields-classifier.git
    cd orkg-research-fields-classifier

Create Virtual Environment:

.. code-block:: bash

    conda create -n <env_name> python=3.8
    conda activate <env_name>

Install all requirements:

.. code-block:: bash

    pip install -r requirements.txt

| Before you can start you need to download some extra data.

| Universal Sentence Encoder:
|   from: https://tfhub.dev/google/universal-sentence-encoder/4
|   to: classifier/models/hub

| ArXiv Dataset:
|   from: https://www.kaggle.com/Cornell-University/arxiv
|   to: classifier/arxiv_data