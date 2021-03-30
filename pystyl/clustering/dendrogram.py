# Hierarchical Agglomerative Cluster Analysis
#
# Copyright (C) 2013 Folgert Karsdorp
# Author: Folgert Karsdorp <fbkarsdorp@gmail.com>
# URL: <https://github.com/fbkarsdorp/HAC-python>
# For licence information, see LICENCE.TXT

import copy
import sys
import os
from operator import itemgetter

import numpy as np
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
from scipy.cluster.hierarchy import to_tree, ClusterNode
import pylab
import matplotlib.pyplot as plt
import seaborn as sns

if sys.version_info[0] == 2:
    from ete2 import Tree, faces, AttrFace, TreeStyle, NodeStyle, TextFace, Face
elif sys.version_info[0] == 3:
    from ete3 import Tree
    from ete3.treeview import AttrFace, NodeStyle, TextFace, TreeStyle, faces, Face

from .. visualization import plt_fig_to_svg

class DendrogramNode(object):
    """
    Represents a node in a dendrogram.
    """
    def __init__(self, id, *children):
        self.id = id
        self.distance = 0.0
        self._children = children

    def leaves(self):
        """
        Return the leaves of all children of a given node.
        """
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


class HashableNode:
    """
    :class:`ClusterNode` is not hashable for some reason, so it won't work
    in ETE. This class adds the necessary functions, based on :data:`id`.
    """
    def __init__(self, cluster_node: ClusterNode):
        self.node = cluster_node

    def __eq__(self, other):
        return self.node.id == other.node.id

    def __hash__(self):
        return hash(self.node.id)

    def __getattr__(self, attr: str):
        return getattr(self.node, attr)


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
        Z = np.array(Z)
        return Z[:,1:]

    def draw_scipy_tree(self, corpus, fontsize=5, color_leafs=True,
                        outputfile=None, save=False, show=True, return_svg=True):
        """
        Draw the dendrogram using plain pylab/scipy/matplotlib.
        """
        plt.clf()
        if outputfile:
            outputfile = os.path.expanduser(outputfile)
        fig = plt.figure()
        ax = fig.add_subplot(111, facecolor='white')
        plt.rcParams['font.family'] = 'arial'
        plt.rcParams['font.size'] = 6
        plt.rcParams['lines.linewidth'] = 0.75
        m = self.to_linkage_matrix()
        labels = corpus.titles
        d = scipy_dendrogram(m, labels=labels,
                             leaf_font_size=fontsize,
                             color_threshold=0.7*max(m[:,2]),
                             leaf_rotation=180)
        ax = plt.gca()
        for idx, label in enumerate(ax.get_xticklabels()):
            label.set_rotation('vertical')
            label.set_fontname('Arial')
            label.set_fontsize(fontsize)
            if color_leafs:
                label.set_color(plt.cm.get_cmap('nipy_spectral')(corpus.target_ints[idx] / 10.))

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
        plt.subplots_adjust(bottom=0.15)
        if save:
            fig.savefig(outputfile)
        if show:
            plt.show()
        if return_svg:
            return plt_fig_to_svg(fig)

    def to_ete(self, labels):
        T = to_tree(self.to_linkage_matrix())
        root = Tree()
        root.dist = 0
        root.name = "root"
        item2node = {HashableNode(T): root}
        to_visit = [T]
        while to_visit:
            node = HashableNode(to_visit.pop())
            cl_dist = node.dist / 2.0
            for ch_node in [node.left, node.right]:
                if ch_node:
                    ch = Tree()
                    ch.dist = cl_dist
                    ch.name = str(ch_node.id)
                    item2node[node].add_child(ch)
                    item2node[HashableNode(ch_node)] = ch
                    to_visit.append(ch_node)

        if labels != None:
            for leaf in root.get_leaves():
                leaf.name = str(labels[int(leaf.name)])
        return root


    def draw_ete_tree(self, corpus, fontsize=5,
                      color_leafs=False,
                      save_newick=True, mode='c',
                      outputfile=None,
                      return_svg=True, show=False,
                      save=False):
        root = self.to_ete(labels=corpus.titles)

        def layout(node):
            if node.is_leaf():
                N = AttrFace("name", fsize=7)
                faces.add_face_to_node(faces.AttrFace("name","Arial",10, None), node, 0, position='branch-right')
                # problems: aligment of labels to branch, left padding of labels

        ts = TreeStyle()
        ts.mode = mode
        ts.show_leaf_name = False
        ts.scale = 120
        ts.show_scale = False
        ts.branch_vertical_margin = 10

        nstyle = NodeStyle()
        nstyle["fgcolor"] = "#0f0f0f"
        nstyle["size"] = 0
        nstyle["vt_line_color"] = "#0f0f0f"
        nstyle["hz_line_color"] = "#0f0f0f"
        nstyle["vt_line_width"] = 1
        nstyle["hz_line_width"] = 1
        nstyle["vt_line_type"] = 0
        nstyle["hz_line_type"] = 0

        for n in root.traverse():
           n.set_style(nstyle)
        
        ts.layout_fn = layout

        if outputfile:
            outputfile = os.path.expanduser(outputfile)

        if save_newick: # save tree in newick format for later manipulation in e.g. FigTree:
            root.write(outfile=os.path.splitext(outputfile)[0]+'.newick')

        if save:
            root.render(outputfile, tree_style=ts)
        if show:
            root.show(tree_style=ts) 
        if return_svg: # return the SVG as a string
            return root.render("%%return")[0]
