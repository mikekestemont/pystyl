# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import glob
import sys
from tempfile import mkdtemp

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
import dendropy
from dendropy.calculate.treesum import TreeSummarizer

if sys.version_info[0] == 2:
    from ete2 import Tree as EteTree
elif sys.version_info[0] == 3:
    from ete3 import Tree as EteTree

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

def distance_matrix(corpus=None, X=None, metric='manhattan'):
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
    - Evert S. et al., Towards a better understanding of Burrows's
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
    if corpus:     
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

def bootstrapped_distance_matrices(corpus, n_iter=100, random_prop=0.50,
              metric='manhattan', random_state=1985):
    dms = []
    try:
        X = corpus.vectorizer.X
        try:
            X = X.toarray()
        except AttributeError:
            pass
    except AttributeError:
        ValueError('Your corpus does not seem to have been vectorized yet.')
    full_size = X.shape[1]
    bootstrap_size = int(full_size*float(random_prop))
    # set random state for replicability:
    np.random.seed(random_state)
    for i in range(n_iter):
        rnd_indices = np.random.randint(low=0, high=full_size, size=bootstrap_size)
        bootstrap_matrix = X[:,rnd_indices]
        dms.append(distance_matrix(X=bootstrap_matrix, metric=metric))
    return dms

def bootstrap_consensus_tree(corpus, trees=[], consensus_level=0.5):
    tmp_dir = mkdtemp()
    for idx, tree in enumerate(trees):
        t = tree.dendrogram.to_ete(labels=corpus.titles)
        t.write(outfile=tmp_dir+'/tree_'+str(idx)+'.newick')
    trees = []
    tns = dendropy.TaxonNamespace(corpus.titles, label="label")
    for filename in glob.glob(tmp_dir+'/*.newick'):
        tree = dendropy.Tree.get(path=filename,
                                 schema='newick',
                                 preserve_underscores=True,
                                 taxon_namespace=tns)
        trees.append(tree)
    
    tsum = TreeSummarizer(support_as_labels=True,
                          support_as_edge_lengths=False,
                          support_as_percentages = True,
                          add_node_metadata = True,
                          weighted_splits = True)
    taxon_namespace = trees[0].taxon_namespace
    split_distribution = dendropy.SplitDistribution(taxon_namespace=taxon_namespace)
    tsum.count_splits_on_trees(trees,
                               split_distribution=split_distribution,
                               is_bipartitions_updated=False)
    tree = tsum.tree_from_splits(split_distribution,
                               min_freq=consensus_level,
                               rooted=False,
                               include_edge_lengths=False) # this param is crucial
    ete_tree = EteTree(tree.as_string("newick").replace('[&U] ', '')+';')
    return ete_tree

