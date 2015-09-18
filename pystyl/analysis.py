from __future__ import print_function

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances

from . distance_metrics import minmax
from . clustering.cluster import VNClusterer, Clusterer

def pca(corpus, nb_dimensions=2):
    """
    Apply dimension reduction to the vectorized
    texts in the corpus, using Principal Components
    Analysis.

    Parameters
    ----------
    corpus : string, default=None
        The corpus to be analyzed.
        Expects that the corpus has been vectorized.
    nb_dimensions : int, default=2
        The nb of components desired. 

    Returns
    ----------
    (pca_matrix, pca_loadings): tuple
        Returns a tuple with:
        - the projection of the corpus texts in
          the reduced space, [n_texts, nb_dimensions]
        - the loadings of the features on each
          component, [n_features, nb_dimensions]

    """
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
    """
    Apply dimension reduction to the vectorized
    texts in the corpus, using t-Distributed
    Stochastic Neighbor Embedding (t-SNE).

    See: L.J.P. van der Maaten and G.E. Hinton.
    Visualizing High-Dimensional Data Using t-SNE.
    Journal of Machine Learning Research 9(Nov):
    2579-2605, 2008.

    Parameters
    ----------
    corpus : string, default=None
        The corpus to be analyzed.
        Expects that the corpus has been vectorized.
    nb_dimensions : int, default=2
        The nb of dimensions in which to project
        the corpus.

    Returns
    ----------
    tsne_matrix : array-like, [n_texts, n_dimensions]
        The projection of the corpus texts in
        the reduced space, [n_texts, nb_dimensions]

    """
    try:
        X = corpus.vectorizer.X
    except AttributeError:
        ValueError('Your corpus does not seem to have been vectorized yet.')
    tsne = TSNE(n_components=nb_dimensions)
    try:
        return tsne.fit_transform(X.toarray()) # unsparsify
    except AttributeError:
        return tsne.fit_transform(X) # input already sparse

def distance_matrix(corpus, metric='manhattan'):
    """
    Calculate a square distance matrix for
    all the texts in the corpus.

    Parameters
    ----------
    corpus : string, default=None
        The corpus to be analyzed.
        Expects that the corpus has been vectorized.
    metric : str, default='manhattan'
        The distance metric to be used for the pairwise
        distance calculations. Currently supports:
        'manhattan', 'cityblock', 'euclidean',
        'cosine', 'minmax'.

    Returns
    ----------
    distance_matrix : 2D-array, [n_texts, n_texts]
        A square distance table holding all the
        pairwise distance calculations.

    Notes:
    ----------
    For a comparison/explication of the metrics consult:
    - S. Argamon, 'Interpreting Burrows's Delta: Geometric
      and Probabilistic Foundations', LLC 23:3 (2008).
    - Evert S. et al., Towards a better understanding of Burrows’s
      Delta in literary authorship attribution. Proceedings of the
      Fourth Workshop on Computational Linguistics for Literature
      (at NAACL HLT 2015), 2015.
    - Koppel et al., Determining if two documents are written by
      the same author, JASIST 2014 (minmax in particular).

    """

    if not metric in ('manhattan', 'cityblock', 'euclidean', 'cosine', 'minmax'):
        raise ValueError('Unsupported distance metric: %s' %(metric))

    # we unsparsify here to make it easier for contributors
    # to add distance functions (in `pystyl.distance_metrics.py`).
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
    """
    Run hierarchical cluster analysis on the texts
    in the corpus.

    Parameters
    ----------
    distance_matrix : 2D-array, [n_texts, n_texts]
        A square distance table holding all the
        pairwise distance calculations.
    linkage : string
        The linkage function to be used in the corpus.

    Returns
    ----------
    cluster : A fitted `Clusterer`

    """
    tree = Clusterer(distance_matrix, linkage=linkage)
    tree.cluster(verbose=0)
    return tree

def vnc_clustering(distance_matrix, linkage):
    """
    Run a  the variability-based neighbor clustering (VNC)
    on the texts in the corpus. The analysis has the property
    that it will respect the order of the texts in the corpus.
    Useful for stylochronometry (or any other application where
    the order of the texts in relevant).
    Will assume that the corpus holds the texts in 
    the correct chronological order.
    Also see: `corpus.temporal_sort()`

    The VNC method been described in e.g.:
    Gries, S. et al., Variability-based neighbor clustering:
    A bottom-up approach to periodization in historical
    linguistics, The Oxford Handbook of the History of
    English, OUP, 2012.

    Parameters
    ----------
    distance_matrix : 2D-array, [n_texts, n_texts]
        A square distance table holding all the
        pairwise distance calculations.
    linkage : string
        The linkage function to be used in the corpus.

    Returns
    ----------
    cluster : A fitted `VNClusterer`

    """
    tree = VNClusterer(distance_matrix, linkage=linkage)
    tree.cluster(verbose=0)
    return tree

