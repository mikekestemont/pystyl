# Hierarchical Agglomerative Cluster Analysis
#
# Copyright (C) 2013 Folgert Karsdorp
# Author: Folgert Karsdorp <fbkarsdorp@gmail.com>
# URL: <https://github.com/fbkarsdorp/HAC-python>
# For licence information, see LICENCE.TXT


class AbstractClusterer(object):
    """
    Abstract interface covering basic clustering functionality.
    """
    def __init__(self, data, linkage, num_clusters):
        """
        @param data: A DistanceMatrix or list of feature value pairs from which 
        a DistanceMatrix can be constructed.
        @type data: L{DistanceMatrix} or C{list}
                
        @param linkage: a clustering of linkage method. The following methods
        are implemented:
            1. Single Linkage (L{single_link})
            2. Complete Linkage (L{complete_link})
            3. Average Linkage (L{average_link})
            4. Median Linkage (L{median_link})
            5. Centroid Linkage (L{centroid_link})
            6. Ward Linkage or Minimum Variance Linkage (L{ward_link})
        @type linkage: C{function}
        
        """
        raise AssertionError('AbstractClusterer is an abstract interface')
        
    def iterate_clusters(self):
        """
        Iterate over all unique vector combinations in the matrix.
        """
        raise AssertionError('AbstractClusterer is an abstract interface')
        
    def smallest_distance(self, clusters):
        """
        Return the smallest distance in the distance matrix.
        The smallest distance depends on the possible connections in the
        distance matrix.
        """
        raise AssertionError('AbstractClusterer is an abstract interface')
        
    def cluster(self, verbose=0, sum_ess=False):
        """
        Cluster all clusters hierarchically unitl the level of 
        num_clusters is obtained.
        """
        raise AssertionError('AbstractClusterer is an abstract interface')
        
    def update_distmatrix(self, i, j, clusters):
        """
        Update the distance matrix using the specified linkage method, so that
        it represents the correct distances to the newly formed cluster.
        """
        return self.linkage(clusters, i, j, self._dendrogram)
        
    def dendrogram(self):
        """
        Return the dendrogram object.
        """
        return self._dendrogram
        
    def num_clusters(self):
        """
        Return the number of clusters.
        """
        return self._num_clusters

