# Hierarchical Agglomerative Cluster Analysis
#
# Copyright (C) 2013 Folgert Karsdorp
# Author: Folgert Karsdorp <fbkarsdorp@gmail.com>
# URL: <https://github.com/fbkarsdorp/HAC-python>
# For licence information, see LICENCE.TXT

import copy
import sys
import numpy as np

import numpy

from operator import itemgetter

class DendrogramNode(object):
    """Represents a node in a dendrogram."""
    def __init__(self, id, *children):
        self.id = id
        self.distance = 0.0
        self._children = children

    def leaves(self):
        """Return the leaves of all children of a given node."""
        if self._children:
            leaves = []
            for child in self._children:
                leaves.extend(child.leaves())
            return leaves
        else:
            return [self]

    def adjacency_list(self):
        """
        For each merge in the dendrogram, return the direct children of
        the cluster, the distance between them and the number of items in the
        cluster (the total number of children all the way down the tree).
        """
        if self._children:
            a_list = [(self.id, self._children[0].id, self._children[1].id,
                       self.distance, len(self))]
            for child in self._children:
                a_list.extend(child.adjacency_list())
            return a_list
        else: return []

    def __len__(self):
        return len(self.leaves())


class Dendrogram(list):
    """
    Class representing a dendrogram. Part is inspired by the Dendrogram class
    of NLTK. It is adjusted to work properly and more efficiently with
    matplotlib and VNC.
    """
    def __init__(self, items):
        super(Dendrogram, self).__init__(map(DendrogramNode, range(len(items))))
        self._num_items = len(self)

    def merge(self, *indices):
        """
        Merge two or more nodes at the given INDICES in the dendrogram.
        The new node will get the index of the first node specified.
        """
        assert len(indices) >= 2
        node = DendrogramNode(
            self._num_items, *[self[i] for i in indices])
        self._num_items += 1
        self[indices[0]] = node
        for i in indices[1:]:
            del self[i]

    def to_linkage_matrix(self):
        Z = self[0].adjacency_list()
        Z.sort()
        Z = numpy.array(Z)
        return Z[:,1:]

    def draw(self, show=True, save=False, format="pdf", labels=None, title=None, fontsize=None):
        """Draw the dendrogram using pylab and matplotlib."""
        try:
            from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
        except ImportError:
            raise ImportError("Scipy not installed, can't draw dendrogram")
        try:
            import pylab
        except ImportError:
            raise ImportError("Pylab not installed, can't draw dendrogram")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib not installed, can't draw dendrogram")

        fig = plt.figure()
        ax = fig.add_subplot(111, axisbg='white')

        plt.rcParams['font.family'] = 'arial'
        plt.rcParams['font.size'] = 6
        plt.rcParams['lines.linewidth'] = 0.75

        m = self.to_linkage_matrix()

        d = scipy_dendrogram(m, labels=labels,
                             leaf_font_size=fontsize,
                             color_threshold=0.7*max(m[:,2]),
                             leaf_rotation=180)

        ax = plt.gca()
        ax_labels = ax.get_xmajorticklabels()+ax.get_ymajorticklabels()
        for i in range(len(ax_labels)):
            ax_labels[i].set_family('arial')

        ax.get_yaxis().set_ticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        plt.xticks(rotation=90)

        plt.tick_params(axis='x', which='both', bottom='off', top='off')
        plt.tick_params(axis='y', which='both', bottom='off', top='off')
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)

        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["savefig.facecolor"] = "white"

        if title is not None:
            fig.suptitle(title, fontsize=12)
        if show:
            fig.show()
        if save:
            fig.savefig('dendrogram.%s' % (format,))

    def ete_tree(self, labels=None):
        if sys.version_info[0] == 2:
            from ete2 import Tree, NodeStyle, TreeStyle
        elif sys.version_info[0] == 3:
            from ete3 import Tree, NodeStyle, TreeStyle
        else:
            raise ValueError('Your version of Python is not supported.')

        from scipy.cluster.hierarchy import to_tree

        T = to_tree(self.to_linkage_matrix())
        root = Tree()
        root.dist = 0
        root.name = "root"
        item2node = {T: root}
        to_visit = [T]
        while to_visit:
            node = to_visit.pop()
            cl_dist = node.dist / 2.0
            for ch_node in [node.left, node.right]:
                if ch_node:
                    ch = Tree()
                    ch.dist = cl_dist
                    ch.name = str(ch_node.id)
                    item2node[node].add_child(ch)
                    item2node[ch_node] = ch
                    to_visit.append(ch_node)
        if labels != None:
            for leaf in root.get_leaves():
                leaf.name = str(labels[int(leaf.name)])

        ts = TreeStyle()
        ts.show_leaf_name = True

        # Draws nodes as small red spheres of diameter equal to 10 pixels
        nstyle = NodeStyle()
        nstyle["shape"] = None
        nstyle["size"] = 0

        # Gray dashed branch lines
        nstyle["hz_line_type"] = 1
        nstyle["hz_line_color"] = "#cccccc"

        # Applies the same static style to all nodes in the tree. Note that,
        # if "nstyle" is modified, changes will affect to all nodes
        for n in root.traverse():
           n.set_style(nstyle)
        return root
