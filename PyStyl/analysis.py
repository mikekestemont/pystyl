from __future__ import print_function

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances

from . distance_metrics import minmax
from . clustering.cluster import VNClusterer, Clusterer

def pca(X, nb_dimensions=2):
    prin_comp = PCA(n_components=nb_dimensions)
    try:
        pca_matrix = prin_comp.fit_transform(X.toarray()) # unsparsify
    except AttributeError:
        pca_matrix = prin_comp.fit_transform(X) # input not sparse

    pca_loadings = prin_comp.components_.transpose()
    return pca_matrix, pca_loadings

def tsne(X, nb_dimensions=2):
    tsne = TSNE(n_components=nb_dimensions)
    try:
        return tsne.fit_transform(X.toarray()) # unsparsify
    except AttributeError:
        return tsne.fit_transform(X) # input already sparse

def distance_matrix(X, metric):
    if not metric in ('manhattan', 'cityblock', 'euclidean', 'cosine', 'minmax'):
        raise ValueError('Unsupported distance metric: %s' %(metric))

    # unsparsify here to make it easier for people to add distance functions...
    try:
        X = X.toarray()
    except AttributeError:
        pass
    if metric == 'minmax':
        return pairwise_distances(X, metric=minmax)
    else:
        return pairwise_distances(X, metric=metric)

def hierarchical_clustering(distance_matrix, linkage):
    cluster_tree = Clusterer(distance_matrix, linkage=linkage)
    cluster_tree.cluster(verbose=0)
    return cluster_tree

