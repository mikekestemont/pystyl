from __future__ import print_function

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca(X, nb_dimensions=2):
    prin_comp = PCA(n_components=nb_dimensions)
    pca_matrix = prin_comp.fit_transform(X.toarray()) # unsparsify
    pca_loadings = prin_comp.components_.transpose()
    return pca_matrix, pca_loadings

def tsne(X, nb_dimensions=2):
    tsne = TSNE(n_components=nb_dimensions)
    return tsne.fit_transform(X.toarray()) # unsparsify

