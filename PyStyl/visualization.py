from __future__ import print_function

from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def scatterplot_2d(X, sample_names, feature_names, sample_categories,
                   outputfile='../output/2d.pdf', nb_clusters=0, loadings=None): 
    sns.set_style('dark')
    sns.plt.rcParams['axes.linewidth'] = 0.4
    fig, ax1 = sns.plt.subplots()  

    # first plot slices:
    x1, x2 = X[:,0], X[:,1]
    ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none')
    if nb_clusters:
        # clustering on top (for colouring):
        clustering = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=nb_clusters)
        clustering.fit(X)
        # add slice names:
        for x, y, name, cluster_label in zip(x1, x2, sample_names, clustering.labels_):
            ax1.text(x, y, name, ha='center', va="center",
                     color=plt.cm.spectral(cluster_label / 10.),
                     fontdict={'family': 'Arial', 'size': 10})
    else:
        for x, y, name, cluster_label in zip(x1, x2, sample_names, sample_categories[0]):
            ax1.text(x, y, name, ha='center', va="center",
                     color=plt.cm.spectral(cluster_label / 10.),
                     fontdict={'family': 'Arial', 'size': 10})

    try:
        # now loadings on twin axis:
        if loadings.any():
            ax2 = ax1.twinx().twiny()
            l1, l2 = loadings[:,0], loadings[:,1]
            ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none');
            for x, y, l in zip(l1, l2, feature_names):
                ax2.text(x, y, l ,ha='center', va="center", size=8, color="darkgrey",
                    fontdict={'family': 'Arial', 'size': 9})
            ax2.set_xticklabels([])
            ax2.set_xticks([])
            ax2.set_yticklabels([])
            ax2.set_yticks([])
    except:
        pass

    # control aesthetics:
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticklabels([])
    ax1.set_yticks([])

    sns.plt.savefig(outputfile, bbox_inches=0)
    plt.clf()
    return


def scatterplot_3d(X, sample_names, sample_categories,
                   outputfile='../output/3d.pdf', nb_clusters=0, loadings=None):
    
    sns.set_style('white')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1, x2, x3 = X[:,0], X[:,1], X[:,2]
    if nb_clusters:
        clustering = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=nb_clusters)
        clustering.fit(X)
        # empty first:
        for x, y, z in zip(x1, x2, x3):
            ax.scatter(x, y, z, edgecolors='none', facecolors='none')
        # add slice names:
        for x, y, z, name, cluster_label in zip(x1, x2, x3, sample_names, clustering.labels_):
            ax.text(x, y, z, name, ha='center', va="center",
                     color=plt.cm.spectral(cluster_label / 10.),
                     fontdict={'family': 'Arial', 'size': 7})
    else:
        for x, y, z in zip(x1, x2, x3):
            ax.scatter(x, y, z, edgecolors='none', facecolors='none')
        for x, y, z, name, cluster_label in zip(x1, x2, x3, sample_names, sample_categories[0]):
            ax.text(x, y, z, name, ha='center', va="center",
                     color=plt.cm.spectral(cluster_label / 10.),
                     fontdict={'family': 'Arial', 'size': 10})

    plt.savefig(outputfile, bbox_inches=0)
    plt.clf()

def clustermap(X, sample_names, sample_categories=None, outputfile='../output/clustermap.pdf', fontsize=5):
    # convert to pandas dataframe:
    df = pd.DataFrame(data=X, columns=(sample_names))
    df = df.applymap(lambda x:int(x*1000)).corr()

    # clustermap plotting:
    cm = sns.clustermap(df)
    ax = cm.ax_heatmap

    # xlabels:
    for idx, label in enumerate(ax.get_xticklabels()):
        label.set_rotation('vertical')
        label.set_fontname('Arial')
        label.set_fontsize(fontsize)
        if sample_categories:
            label.set_color(plt.cm.spectral(sample_categories[0][idx] / 10.))

    # ylabels:
    for idx, label in enumerate(ax.get_yticklabels()):
        label.set_rotation('horizontal')
        label.set_fontname('Arial')
        label.set_fontsize(fontsize)
        if sample_categories:
            label.set_color(plt.cm.spectral(sample_categories[0][-idx-1] / 10.)) # watch out: different indexing both axis

    cm.savefig(outputfile)
    plt.clf()

