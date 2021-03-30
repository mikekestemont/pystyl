# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from bokeh.models import HoverTool, ColumnDataSource, Axis
from bokeh.plotting import figure, show, output_file, save

if sys.version_info[0] == 2:
    from ete2 import Tree, faces, AttrFace, TreeStyle, NodeStyle, TextFace, Face
elif sys.version_info[0] == 3:
    from ete3 import Tree
    from ete3.treeview import AttrFace, NodeStyle, TextFace, TreeStyle, faces, Face

def plt_fig_to_svg(fig):
    imgdata = StringIO.StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)  # rewind the data
    return imgdata.buf  # this is svg data

def scatterplot(corpus, plot_type='static', nb_clusters=0,
                coor=None, loadings=None, outputfile=None,
                save=False, show=False, return_svg=False):
    """
    Draw two-dimensional scatterplot of corpus, given the
    coordinates passed.
    
    Parameters
    ----------
    plot_type : string, default='static'
        The type of plot to be made. Currently supports:
        - 'static': will call `static_scatterplot()`
        - 'interactive': will call `interactive_scatterplot()`

    All other parameters are passed to `static_scatterplot()`
    and `interactive_scatterplot()`.

    """
    if outputfile:
        outputfile = os.path.expanduser(outputfile)
    if plot_type == 'static':
        return static_scatterplot(corpus, coor, loadings=loadings,
                                  outputfile=outputfile, nb_clusters=nb_clusters,
                                  save=save, show=show, return_svg=return_svg)
    elif plot_type == 'interactive':
        return interactive_scatterplot(corpus=corpus, coor=coor,
                                  outputfile=outputfile, nb_clusters=nb_clusters,
                                  save=save, show=show, return_svg=return_svg)
    else:
        raise ValueError('Unsupported plot_type: %s' %(plot_type))

def static_scatterplot(corpus, coor=None, outputfile=None,
                       nb_clusters=0, loadings=None,
                       save=False, show=False, return_svg=False):
    """
    Draw two-dimensional scatterplot of the corpus, given the
    coordinates passed. Produces a static matplotlib/seaborn
    plot.
    
    Parameters
    ----------
    corpus : `Corpus` instance
        The corpus to be plotted.
    coor : array-like, [n_texts, 2]
        The coordinates of the texts to be used.
    outputfile : str
        The path where the plot should be saved.
    nb_clusters : int, default=0
        If `nb_clusters` > 0, will run a hierarchical
        cluster analysis, identifying `nb_clusters`
        clusters using `coor`. Texts will be coloured
        according to these clusters as a reading aid.
        Else, clusters will be colored according to
        their category.
    loadings : array-like [n_features, n_dimensions]
        If loadings are passed, the scatterplot will
        be overlaid with the loadings of the original
        features on the twin axis.
    save : boolean, default=False
        Whether to save the plot to `outputfile`.


    """
    plt.clf()
    if coor.shape[1] < 2:
        raise ValueError('Only two-dimensional matrices are supported')
    if coor is None:
        ValueError('Please specify valid (2D) coordinates')

    sns.set_style('dark')
    plt.rcParams['axes.linewidth'] = 0.4
    fig, ax1 = plt.subplots()  

    labels = corpus.titles
    # first plot slices:
    x1, x2 = coor[:,0], coor[:,1]
    ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none')
    if nb_clusters:
        # clustering on top (for colouring):
        clustering = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=nb_clusters)
        clustering.fit(coor)
        # add slice names:
        for x, y, name, cluster_label in zip(x1, x2, labels, clustering.labels_):
            ax1.text(x, y, name, ha='center', va="center",
                     color=plt.cm.get_cmap('nipy_spectral')(cluster_label / 10.),
                     fontdict={'family': 'Arial', 'size': 10})
    else:
        for x, y, name, cluster_label in zip(x1, x2, labels, corpus.target_ints):
            ax1.text(x, y, name, ha='center', va="center",
                     color=plt.cm.get_cmap('nipy_spectral')(cluster_label / 10.),
                     fontdict={'family': 'Arial', 'size': 10})
    try:
        # now loadings on twin axis:
        if loadings.any():
            ax2 = ax1.twinx().twiny()
            l1, l2 = loadings[:,0], loadings[:,1]
            ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none');
            for x, y, l in zip(l1, l2, corpus.vectorizer.feature_names):
                ax2.text(x, y, l ,ha='center', va="center", size=8, color="darkgrey",
                    fontdict={'family': 'Arial', 'size': 9})
            ax2.set_xticklabels([])
            ax2.set_xticks([])
            ax2.set_yticklabels([])
            ax2.set_yticks([])
    except AttributeError:
        pass

    # control aesthetics:
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    if save:
        plt.savefig(outputfile, bbox_inches=0)
    if show:
        plt.show()
    if return_svg:
        return plt_fig_to_svg(fig)

    return

def interactive_scatterplot(corpus=None, coor=None, outputfile=None, nb_clusters=0,
                            save=False, show=False, return_svg=False):
    """
    Draw an interactive two-dimensional html-scatterplot
    of the corpus, given the coordinates passed. Uses Bokeh
    to produce an interactive plots which supports hovering.
    Useful for visualizing large datasets without cluttering
    the plot with sample names.
    
    Parameters
    ----------
    corpus : `Corpus` instance
        The corpus to be plotted.
    coor : array-like, [n_texts, 2]
        The coordinates of the texts to be used.
    outputfile : str
        The path where the plot should be saved.
    nb_clusters : int, default=0
        If `nb_clusters` > 0, will run a hierarchical
        cluster analysis, identifying `nb_clusters`
        clusters using `coor`. Texts will be coloured
        according to these clusters as a reading aid.
        Else, clusters will be colored according to
        their category.
    save : boolean, default=False
        Whether to save the plot to `outputfile`.

    """
    if coor.shape[1] < 2:
        raise ValueError('Only two-dimensional matrices are supported')

    output_file(outputfile)
    TOOLS="pan,wheel_zoom,reset,hover,box_select,save"
    p = figure(tools=TOOLS,
               plot_width=800, title_text_font="Arial", 
               plot_height=600, outline_line_color="white")

    if nb_clusters:
        cl = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=nb_clusters)
        clusters = cl.fit_predict(coor)
        # get color palette:
        colors = sns.color_palette('husl', n_colors=nb_clusters)
        colors = [tuple([c * 256 for c in color]) for color in colors]
        colors = ['#%02x%02x%02x' % colors[i] for i in clusters]
    else:
        colors = sns.color_palette('husl', n_colors=len(corpus.target_idx))
        colors = [tuple([c * 256 for c in color]) for color in colors]
        colors = ['#%02x%02x%02x' % colors[i] for i in corpus.target_ints]

    labels = corpus.titles
    
    source = ColumnDataSource(data=dict(x=coor[:,0], y=coor[:,1], name=labels))
    
    p.circle(x=coor[:,0], y=coor[:,1],
             source=source, size=8, color=colors,
             fill_alpha=0.9, line_color=None)

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [("name", "@name")]
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_label_text_font_size = '0pt'
    p.axis.major_tick_line_color = None
    p.axis[0].ticker.num_minor_ticks = 0
    p.axis[1].ticker.num_minor_ticks = 0
    if save:
        save(p)


def scatterplot_3d(corpus, coor, outputfile,
                  nb_clusters=0,
                  save=False, show=False, return_svg=False):
    """
    Draw a 3-dimensional scatterplot of the corpus, given the
    coordinates passed. Produces a static matplotlib/seaborn
    plot.
    
    Parameters
    ----------
    corpus : `Corpus` instance
        The corpus to be plotted.
    coor : array-like, [n_texts, 3]
        The coordinates of the texts to be used.
    outputfile : str
        The path where the plot should be saved.
    nb_clusters : int, default=0
        If `nb_clusters` > 0, will run a hierarchical
        cluster analysis, identifying `nb_clusters`
        clusters using `coor`. Texts will be coloured
        according to these clusters as a reading aid.
        Else, clusters will be colored according to
        their category.
    save : boolean, default=False
        Whether to save the plot to `outputfile`.

    """
    plt.clf()
    if coor.shape[1] < 3:
        raise ValueError('Only three-dimensional matrices are supported')
    if not outputfile:
        outputfile = os.path.expanduser(outputfile)
    
    sns.set_style('white')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1, x2, x3 = coor[:,0], coor[:,1], coor[:,2]
    labels = corpus.titles
    if nb_clusters:
        clustering = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=nb_clusters)
        clustering.fit(coor)
        # empty first:
        for x, y, z in zip(x1, x2, x3):
            ax.scatter(x, y, z, edgecolors='none', facecolors='none')
        # add slice names:
        for x, y, z, name, cluster_label in zip(x1, x2, x3, labels, clustering.labels_):
            ax.text(x, y, z, name, ha='center', va="center",
                     color=plt.cm.get_cmap('nipy_spectral')(cluster_label / 10.),
                     fontdict={'family': 'Arial', 'size': 7})
    else:
        for x, y, z in zip(x1, x2, x3):
            ax.scatter(x, y, z, edgecolors='none', facecolors='none')
        for x, y, z, name, cluster_label in zip(x1, x2, x3, labels, corpus.target_ints):
            ax.text(x, y, z, name, ha='center', va="center",
                     color=plt.cm.get_cmap('nipy_spectral')(cluster_label / 10.),
                     fontdict={'family': 'Arial', 'size': 10})
    if save:
        plt.savefig(outputfile, bbox_inches=0)
    if show:
        plt.show()
    if return_svg:
        return plt_fig_to_svg(fig)


def clustermap(corpus, distance_matrix=None, color_leafs=True,
               outputfile=None, fontsize=5,
               save=False, show=False, return_svg=False):
    """
    Draw a square clustermap of the corpus using seaborn's
    `clustermap`.
    
    Parameters
    ----------
    corpus : `Corpus` instance
        The corpus to be plotted.
    distance_matrix : array-like, [n_texts, n_texts]
        A square distance matrix holding the 
        pairwise distances between all the texts in 
        the corpus.
    color_leafs: boolean, default=True,
        If true, will color the text labels on the
        axis according to their category.
    outputfile : str
        The path where the plot should be saved.
    fontsize : int, default=5
        The fontsize of the labels on the axes.
    save : boolean, default=False
        Whether to save the plot to `outputfile`.

    """
    plt.clf()
    # convert to pandas dataframe:
    labels = corpus.titles
    df = pd.DataFrame(data=distance_matrix, columns=labels)
    df = df.applymap(lambda x:int(x*1000)).corr()

    # clustermap plotting:
    cm = sns.clustermap(df)
    ax = cm.ax_heatmap
        # xlabels:
    for idx, label in enumerate(ax.get_xticklabels()):
        label.set_rotation('vertical')
        label.set_fontname('Arial')
        label.set_fontsize(fontsize)
        if color_leafs:
            label.set_color(plt.cm.get_cmap('nipy_spectral')(corpus.target_ints[idx] / 10.))

    # ylabels:
    for idx, label in enumerate(ax.get_yticklabels()):
        label.set_rotation('horizontal')
        label.set_fontname('Arial')
        label.set_fontsize(fontsize)
        if color_leafs:
            label.set_color(plt.cm.get_cmap('nipy_spectral')(corpus.target_ints[-idx-1] / 10.)) # watch out: different indexing on this axis
    if save:
        if outputfile:
            outputfile = os.path.expanduser(outputfile)
        cm.savefig(outputfile)
    if show:
        plt.show()
    if return_svg:
        return plt_fig_to_svg(cm)
    

def scipy_dendrogram(corpus, tree, outputfile=None,
                     fontsize=5, color_leafs=True,
                     show=True, save=False, return_svg=True):
    """
    Draw a dendrogram of the texts in the corpus using scipy.
    
    Parameters
    ----------
    corpus : `Corpus` instance
        The corpus to be plotted.
    tree : `(VN)Clusterer` object
        The clusterer object which was
        applied to the corpus.
    color_leafs: boolean, default=True,
        If true, will color the text labels
        according to their category.
    outputfile : str
        The path where the plot should be saved.
    fontsize : int, default=5
        The fontsize of the labels on the axes.

    """
    return tree.dendrogram.draw_scipy_tree(corpus, outputfile=outputfile,
                  fontsize=fontsize, color_leafs=color_leafs, save=save,
                  show=show, return_svg=return_svg)

def ete_dendrogram(corpus, tree, outputfile=None,
                   fontsize=5, save_newick=True, mode='c', show=False,
                   color_leafs=False, save=False, return_svg=True):
    """
    Draw a dendrogram of the texts in the corpus using ETE.
    
    Parameters
    ----------
    corpus : `Corpus` instance
        The corpus to be plotted.
    tree : `(VN)Clusterer` object
        The clusterer object which was
        applied to the corpus.
    outputfile : str
        The path where the plot should be saved.
    color_leafs: boolean, default=True,
        If true, will color the text labels
        according to their category.
    fontsize : int, default=5
        The fontsize of the labels on the axes.
    save_newick : boolean, default=True
        Whether to dump a representation of the
        tree in newick-format, which can later
        be modified using software like FigTree:
        http://tree.bio.ed.ac.uk/software/figtree/
    mode : str, default='c'
        The type of tree to be drawn. Supports:
        - 'c': circular dendrogram
        - 'r': traditional, rectangular dendrogram
    save : boolean, default=False
        Whether to save the plot to `outputfile`.
    return_svg : boolean, default=True
        Whether to return the plot in SVG-format.
        Useful for the GUI.
    """
    return tree.dendrogram.draw_ete_tree(corpus, outputfile=outputfile,
                   fontsize=fontsize, save_newick=save_newick, mode=mode,
                   color_leafs=color_leafs,
                   save=save, return_svg=return_svg, show=show)

def bct_dendrogram(corpus, tree, outputfile=None,
                   fontsize=5, save_newick=True, mode='c', show=False,
                   color_leafs=False, save=False, return_svg=True):
    """
    Draw a dendrogram of the texts in the corpus using ETE.
    
    Parameters
    ----------
    corpus : `Corpus` instance
        The corpus to be plotted.
    tree : `(VN)Clusterer` object
        The clusterer object which was
        applied to the corpus.
    outputfile : str
        The path where the plot should be saved.
    color_leafs: boolean, default=True,
        If true, will color the text labels
        according to their category.
    fontsize : int, default=5
        The fontsize of the labels on the axes.
    save_newick : boolean, default=True
        Whether to dump a representation of the
        tree in newick-format, which can later
        be modified using software like FigTree:
        http://tree.bio.ed.ac.uk/software/figtree/
    mode : str, default='c'
        The type of tree to be drawn. Supports:
        - 'c': circular dendrogram
        - 'r': traditional, rectangular dendrogram
    save : boolean, default=False
        Whether to save the plot to `outputfile`.
    return_svg : boolean, default=True
        Whether to return the plot in SVG-format.
        Useful for the GUI.
    """

    for leaf in tree.get_leaves():
        leaf.name = leaf.name.replace("'", '')
    
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

    for n in tree.traverse():
       n.set_style(nstyle)
        
    ts.layout_fn = layout

    if outputfile:
        outputfile = os.path.expanduser(outputfile)

    if save_newick: # save tree in newick format for later manipulation in e.g. FigTree:
        tree.write(outfile=os.path.splitext(outputfile)[0]+'.newick')

    if save:
        tree.render(outputfile, tree_style=ts)
    if show:
        tree.show(tree_style=ts) 
    if return_svg: # return the SVG as a string
        return tree.render("%%return")[0]
        


