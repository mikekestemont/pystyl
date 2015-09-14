# Hierarchical Agglomerative Cluster Analysis
#
# Copyright (C) 2013 Folgert Karsdorp
# Author: Folgert Karsdorp <fbkarsdorp@gmail.com>
# URL: <https://github.com/fbkarsdorp/HAC-python>
# For licence information, see LICENCE.TXT

from __future__ import division
import numpy as np

def _general_link(clusters, i, j, method):
    """
    This function is used to update the distance matrix in the clustering
    procedure.

    Several linkage methods for hierarchical agglomerative clustering
    can be used: 
        - single linkage; 
        - complete linkage;
        - group average linkage;
        - median linkage; 
        - centroid linkage and 
        - ward linkage.
    
    All linkage methods use the Lance-Williams update formula:
    M{d(ij,k) = S{alpha}(i)*d(i,k) + S{alpha}(j)*d(j,k) + S{beta}*d(i,j) + 
    S{gamma}*(d(i,k) - d(j,k))}
    
    In the functions below, the following symbols represent the parameters in
    the update formula:
        1. n_x = length cluster
        2. a_x = S{alpha}(x)
        3. b_x = S{beta}(x)
        4. c_x = S{gamma}(x)
        5. d_xy = distance(x,y) = d(x,y)
        
    @param clusters: an object of the class L{DistanceMatrix}
    @type clusters: L{DistanceMatrix}
    
    @param i: cluster A 
    @type i: C{int}
    
    @param j: cluster B
    @type j: C{int}
    
    @param method: the method used for clustering.
    @type method: a function
    
    @return: an updated distance matrix
    """
    for k in range(len(clusters)):
        if k != i and k != j:
            if method.__name__ == "ward_update":
                new_distance = method(clusters[i,k], clusters[j,k], k)
            else:
                new_distance = method(clusters[i,k], clusters[j,k])
            clusters[i,k] = new_distance
            clusters[k,i] = new_distance
    return clusters

def single_link(clusters, i, j, dendrogram):
    """
    Hierarchical Agglomerative Clustering using single linkage. Cluster j is
    clustered with cluster i when the minimum distance between any
    of the members of i and j is the smallest distance in the vector space.
    
    Lance-Williams parameters:
    
    M{S{alpha}(i) = 0.5; S{beta} = 0; S{gamma} = -0.5} which equals
    M{min(d(i,k),d(j,k))}
    """
    ks = np.arange(clusters.shape[0])
    ks = ks[(ks!=i) & (ks!=j)]
    minima = np.minimum(clusters[i,], clusters[j,])[ks]
    clusters[i,ks] = minima
    clusters[ks,i] = minima
    return clusters
    # return _general_link(clusters, i, j, min)

def complete_link(clusters, i, j, dendrogram):
    """
    Hierarchical Agglomerative Clustering using complete linkage. Cluster j is
    clustered with cluster i when the maximum distance between any
    of the members of i and j is the smallest distance in the vector space.

    Lance-Williams parameters:
    
    M{S{alpha}(i) = 0.5; S{beta} = 0; S{gamma} = 0.5} which equals 
    M{max(d(i,k),d(j,k))}
    """
    ks = np.arange(clusters.shape[0])
    ks = ks[(ks!=i) & (ks!=j)]
    maxima = np.maximum(clusters[i,], clusters[j,])[ks]
    clusters[i,ks] = maxima
    clusters[ks,i] = maxima
    return clusters
    # return _general_link(clusters, i, j, max)

def average_link(clusters, i, j, dendrogram):
    """
    Hierarchical Agglomerative Clustering using group average linkage. Cluster j
    is clustered with cluster i when the pairwise average of values between the
    clusters is the smallest in the vector space.
    
    Lance-Williams parameters:
    
    M{S{alpha}(i) = |i|/(|i|+|j|); S{beta} = 0; S{gamma} = 0}
    """
    n_i, n_j = len(dendrogram[i]), len(dendrogram[j])
    a_i = n_i / (n_i + n_j)
    a_j = n_j / (n_i + n_j)
    update_fn = lambda d_ik,d_jk: a_i*d_ik + a_j*d_jk
    return _general_link(clusters, i, j, update_fn)

def median_link(clusters, i, j, dendrogram):
    """
    Hierarchical Agglomerative Clustering using median linkage. Cluster j
    is clustered with cluster i when the distance between the median values
    of the clusters is the smallest in the vector space.
    
    Lance-Williams parameters:
    
    M{S{alpha}(i) = 0.5; S{beta} = -0.25; S{gamma} = 0}
    """
    update_fn = lambda d_ik,d_jk: 0.5*d_ik + 0.5*d_jk + -0.25*clusters[i,j]
    return _general_link(clusters, i, j, update_fn)

def centroid_link(clusters, i, j, dendrogram):
    """
    Hierarchical Agglomerative Clustering using centroid linkage. Cluster j
    is clustered with cluster i when the distance between the centroids of the
    clusters is the smallest in the vector space.
    
    Lance-Williams parameters:
    
    M{S{alpha}(i) = |i| / (|i| + |j|); S{beta} = -|i||j| / (|i|+ |j|)^2; 
    S{gamma} = 0}
    """
    n_i, n_j = len(dendrogram[i]), len(dendrogram[j])
    a_i = n_i / (n_i + n_j)
    a_j = n_j / (n_i + n_j)
    b = -(n_i * n_j) / (n_i + n_j)**2
    update_fn = lambda d_ik,d_jk: a_i*d_ik + a_j*d_jk + b*clusters[i,j]
    return _general_link(clusters, i, j, update_fn)

def ward_link(clusters, i, j, dendrogram):
    """
    Hierarchical Agglomerative Clustering using Ward's linkage. Two clusters i
    and j are merged when their merge results in the smallest increase in the
    sum of error squares in the vector space.
    
    Lance-Williams parameters:
    
    M{S{alpha}(i) = (|i| + |k|) / (|i| + |j| + |k|); 
    S{beta} = -|k|/(|i| + |j| + |k|); S{gamma} = 0}
    """
    n_i, n_j = len(dendrogram[i]), len(dendrogram[j])
    def ward_update(d_ik, d_jk, k):
        n_k = len(dendrogram[k])
        n_ijk = n_i+n_j+n_k
        return ( (n_i+n_k)/(n_ijk)*d_ik + (n_j+n_k)/(n_ijk)*d_jk +
                 -(n_k/(n_ijk))*clusters[i][j] )
    return _general_link(clusters, i, j, ward_update)

LINKAGES = {'ward': ward_link, 
            'complete': complete_link,
            'single': single_link,
            'centroid': centroid_link,
            'average': average_link,
            'median': median_link}

def linkage_fn(linkage):
    if linkage in LINKAGES:
        return LINKAGES[linkage]
    raise ValueError("Linkage funtion '%s' is not supported" % linkage)

__all__ = ['single_link', 'complete_link', 'centroid_link', 'ward_link',
           'median_link', 'average_link']
