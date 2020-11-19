import codecs
from os import path

from hazm import POSTagger
import pandas as pd

_data_path = path.join(path.dirname(__file__), 'data/')
_external_data_path = path.join(_data_path, 'external/')
_stoplists_path = path.join(_data_path, 'stoplist/')


class ExternalList():
    punctuations = path.join(_external_data_path, 'puncs.dat')
    shifters = path.join(_external_data_path, 'shifters')


class StopLists():
    stop_hazm = path.join(_stoplists_path, 'hazm_stopwords.dat')
    stop_post_verbs = path.join(_stoplists_path, 'c')
    stop_post_puncs = path.join(_stoplists_path, 'post-stop-puncs')
    stop_start_words = path.join(_stoplists_path, 'startstops')
    stop_after_words = path.join(_stoplists_path, 'afterkeystop')


class ListReader():
    def __init__(self, list):
        self.list = list

    def get(self):
        with codecs.open(self.list, encoding='utf8') as list_file:
            return list(set(map(lambda w: w.strip(), list_file)))


class StopListReader(ListReader):
    def __init__(self, stop_list):
        super(StopListReader, self).__init__(stop_list)


class ExternalListReader(ListReader):
    def __init__(self, external_list):
        super(ExternalListReader, self).__init__(external_list)


def getPersianVerbs_dataset():
    alldf = pd.read_excel(_external_data_path + 'persian_verbs.xls')
    alldf.columns = ['mazi', 'mozare', 'pishvand', 'felyar', 'harfezafe', 'sakht']

    return list(alldf.mazi) + list(alldf.mozare)


def getPosTagger():
    tagger = POSTagger(model=_external_data_path + 'postagger.model')
    return tagger


if __name__ == "__main__":
    a = StopListReader(StopLists.stop_after_words).get()
    b = ExternalListReader(ExternalList.punctuations).get()
    print(b)
