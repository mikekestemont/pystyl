from __future__ import print_function

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances

from . distance_metrics import minmax
from . clustering.cluster import VNClusterer, Clusterer

def pca(corpus, nb_dimensions=2):
    try:
        X = corpus.vectorizer.X
    except AttributeError:
        ValueError('Your corpus does not seem to have been vectorized yet.')
    prin_comp = PCA(n_components=nb_dimensions)
    try:
        pca_matrix = prin_comp.fit_transform(X.toarray()) # unsparsify
    except AttributeError:
        pca_matrix = prin_comp.fit_transform(X) # input not sparse

    pca_loadings = prin_comp.components_.transpose()
    return pca_matrix, pca_loadings

def tsne(corpus, nb_dimensions=2):
    try:
        X = corpus.vectorizer.X
    except AttributeError:
        ValueError('Your corpus does not seem to have been vectorized yet.')
    tsne = TSNE(n_components=nb_dimensions)
    try:
        return tsne.fit_transform(X.toarray()) # unsparsify
    except AttributeError:
        return tsne.fit_transform(X) # input already sparse

def distance_matrix(corpus, metric):
    if not metric in ('manhattan', 'cityblock', 'euclidean', 'cosine', 'minmax'):
        raise ValueError('Unsupported distance metric: %s' %(metric))

    # unsparsify here to make it easier for people to add distance functions...
    try:
        X = corpus.vectorizer.X
        try:
            X = X.toarray()
        except AttributeError:
            pass
    except AttributeError:
        ValueError('Your corpus does not seem to have been vectorized yet.')

    if metric == 'minmax':
        return pairwise_distances(X, metric=minmax)
    else:
        return pairwise_distances(X, metric=metric)

def hierarchical_clustering(distance_matrix, linkage):
    tree = Clusterer(distance_matrix, linkage=linkage)
    tree.cluster(verbose=0)
    return tree

def vnc_clustering(distance_matrix, linkage):
    tree = VNClusterer(distance_matrix, linkage=linkage)
    tree.cluster(verbose=0)
    return tree

