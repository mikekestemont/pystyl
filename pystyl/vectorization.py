from __future__ import print_function

import os
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
elif sys.version_info[0] == 3:
    import pickle

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.sparse as sp
import numpy as np

std_output_path = os.path.dirname(os.path.abspath(__file__))+'/../output/'

def identity(y):
    return y

class StdDevScaler(BaseEstimator):

    def fit(self, X, y=None):
        self.weights = StandardScaler(with_mean=False).fit(X).std_
        return self

    def transform(self, X):
        X = sp.csr_matrix(X, dtype=np.float64)
        for i in range(X.shape[0]):
            start, end = X.indptr[i], X.indptr[i+1]
            X.data[start:end] /= self.weights[X.indices[start:end]]
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class Vectorizer:

    def __init__(self, mfi=100, ngram_type='word',
                 ngram_size=1, vocabulary=None,
                 vector_space='tf', lowercase=True,
                 min_df=0.0, max_df=1.0, ignore=None):

        if vector_space not in ('tf', 'tf_scaled', 'tf_std', 'tf_idf', 'bin'):
            raise ValueError('Wrong vector vector space model: %s' %(vector_space))

        self.params = {'max_features': mfi,
                 'max_df': max_df,
                 'min_df': min_df,
                 'preprocessor': None,
                 'ngram_range': (ngram_size, ngram_size),
                 'lowercase': False,
                 'decode_error': 'ignore',
                }

        if ngram_type == 'word':
            self.params['tokenizer'] = identity
        elif ngram_type in ('char', 'char_wb'):
            self.params['analyzer'] = ngram_type

        if vocabulary:
            self.params['vocabulary'] = vocabulary

        if vector_space == 'tf':
            self.params['use_idf'] = False
            v = TfidfVectorizer(**self.params)
            self.transformer = Pipeline([('s1', v)])

        elif vector_space == 'tf_scaled':
            self.params['use_idf'] = False
            v = TfidfVectorizer(**self.params)
            scaler = StandardScaler(with_mean=False)
            self.transformer = Pipeline([('s1', v), ('s2', scaler)])

        elif vector_space == 'tf_std':
            self.params['use_idf'] = False
            v = TfidfVectorizer(**self.params)
            scaler = StdDevScaler()
            self.transformer = Pipeline([('s1', v), ('s2', scaler)])

        elif vector_space == 'tf_idf':
            self.params['use_idf'] = True
            v = TfidfVectorizer(**self.params)
            self.transformer = Pipeline([('s1', v)])

        elif vector_space == 'bin':
            self.params['binary'] = True
            v = CountVectorizer(**self.params)
            self.transformer = Pipeline([('s1', v)])

    def vectorize(self, texts):
        self.X = self.transformer.fit_transform(texts)
        # extract names for later convenience:
        self.feature_names = self.transformer.named_steps['s1'].get_feature_names()
        return self.X


