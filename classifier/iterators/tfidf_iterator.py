import dask.dataframe as dd
from gensim.corpora import Dictionary
from typing import List


class TfidfIterator:
    """ Iterator that iterates over all documents in Dataframe and builds bow Dictionary for tfidf"""

    def __init__(self, source_dd: dd.DataFrame, dictionary: Dictionary):
        self.source_dd = source_dd
        self.dictionary = dictionary

    def __iter__(self) -> List:
        """ Iterate over all rows in dask dataframe and yield bow representation of statements."""
        for line in self.source_dd.iterrows():

            yield self.dictionary.doc2bow(line[1].values[0].split(), allow_update=True)
