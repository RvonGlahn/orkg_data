## ORKG Data Processing and Classification

### This project is about two  tasks:

*  collecting and processing missing ORKG paper meta data
*  building a classifier for the ORKG Research Fields

### Installation


Install from gitlab repository:

```
git clone https://gitlab.com/TIBHannover/orkg/orkg-research-fields-classifier.git
cd orkg-research-fields-classifier
```
Create Virtual Environment:

```
conda create --name ResearchField python=3.8
conda activate ResearchField
pip install -r requirements.txt
```

You find the most important info in run_script.py.
```
python run_script.py
```


Before you can start you need to download some extra data.

Universal Sentence Encoder:
- from: https://tfhub.dev/google/universal-sentence-encoder/4
- to: classifier/models/hub

 ArXiv Dataset:
 - from: https://www.kaggle.com/Cornell-University/arxiv
 - to: classifier/arxiv_data

### Documentation

For further information read the docs: https://orkg-data-collection.readthedocs.io/en/latest/