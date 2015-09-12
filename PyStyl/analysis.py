from __future__ import print_function

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def pca(X, sample_names, feature_names, sample_categories,
        outputfile='../output/pca.pdf', nb_dimensions=2, nb_clusters=0,
        loadings=True):
    """
    Run pca on matrix and visualize samples in 1st 2 PCs, with word loadings projected
    on top. If nb_clusters is None, the coloring of samples depends on their category.
    Else, the colouring of the samples is provided by running a cluster analysis
    on the samples in these first dimensions. 
    """  

    prin_comp = PCA(n_components=nb_dimensions)
    pca_matrix = prin_comp.fit_transform(X.toarray()) # unsparsify
    pca_loadings = prin_comp.components_.transpose()
    
    if nb_dimensions == 2:
        sns.set_style('dark')
        sns.plt.rcParams['axes.linewidth'] = 0.4
        fig, ax1 = sns.plt.subplots()  
        # first plot slices:
        x1, x2 = pca_matrix[:,0], pca_matrix[:,1]
        ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none')

        if nb_clusters:
            # clustering on top (for colouring):
            clustering = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=nb_clusters)
            clustering.fit(pca_matrix)

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

        # now loadings on twin axis:
        if loadings:
            ax2 = ax1.twinx().twiny()
            l1, l2 = pca_loadings[:,0], pca_loadings[:,1]
            ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none');

            for x, y, l in zip(l1, l2, feature_names):
                ax2.text(x, y, l ,ha='center', va="center", size=8, color="darkgrey",
                    fontdict={'family': 'Arial', 'size': 9})

        # control aesthetics:
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.set_xticklabels([])
        ax1.set_xticks([])
        ax1.set_yticklabels([])
        ax1.set_yticks([])
        ax2.set_xticklabels([])
        ax2.set_xticks([])
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        sns.plt.tight_layout()
        sns.plt.savefig(outputfile, bbox_inches=0)
        plt.clf()

    elif nb_dimensions == 3:
        sns.set_style('white')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x1, x2, x3 = pca_matrix[:,0], pca_matrix[:,1], pca_matrix[:,2]

        if nb_clusters:
            # clustering on top (for colouring):
            clustering = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=nb_clusters)
            clustering.fit(pca_matrix)

            # empty first:
            for x, y, z, name, cluster_label in zip(x1, x2, x3, sample_names, clustering.labels_):
                ax.scatter(x, y, z, edgecolors='none', facecolors='none')
            # add slice names:
            for x, y, z, name, cluster_label in zip(x1, x2, x3, sample_names, clustering.labels_):
                ax.text(x, y, z, name, ha='center', va="center",
                         color=plt.cm.spectral(cluster_label / 10.),
                         fontdict={'family': 'Arial', 'size': 7})

        else:
            for x, y, z, name, cluster_label in zip(x1, x2, x3, sample_names, sample_categories[0]):
                ax.text(x, y, z, name, ha='center', va="center",
                         color=plt.cm.spectral(cluster_label / 10.),
                         fontdict={'family': 'Arial', 'size': 10})
        """
        # now loadings on twin axis:
        if loadings:
            ax2 = ax1.twinx().twiny()
            l1, l2 = pca_loadings[:,0], pca_loadings[:,1]
            ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none');

            for x, y, l in zip(l1, l2, feature_names):
                ax2.text(x, y, l ,ha='center', va="center", size=8, color="darkgrey",
                    fontdict={'family': 'Arial', 'size': 9})
        """
        plt.savefig(outputfile, bbox_inches=0)
        plt.clf()


def tsne(X, sample_names, sample_categories,
        outputfile='../output/tsne.pdf', nb_dimensions=2, nb_clusters=0,
        loadings=True):
    """
    Run pca on matrix and visualize samples in 1st 2 PCs, with word loadings projected
    on top. If nb_clusters is None, the coloring of samples depends on their category.
    Else, the colouring of the samples is provided by running a cluster analysis
    on the samples in these first dimensions. 
    """
    sns.set_style('dark')
    sns.plt.rcParams['axes.linewidth'] = 0.4
    fig, ax1 = sns.plt.subplots()    

    tsne = TSNE(n_components=nb_dimensions)
    tsne_matrix = tsne.fit_transform(X.toarray()) # unsparsify
    
    # first plot slices:
    x1, x2 = tsne_matrix[:,0], tsne_matrix[:,1]
    ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none')

    if nb_clusters:
        # clustering on top (for colouring):
        clustering = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=nb_clusters)
        clustering.fit(pca_matrix)

        # add slice names:
        for x, y, name, cluster_label in zip(x1, x2, sample_names, clustering.labels_):
            ax1.text(x, y, name.split, ha='center', va="center",
                     color=plt.cm.spectral(cluster_label / 10.),
                     fontdict={'family': 'Arial', 'size': 10})

    else:
        for x, y, name, cluster_label in zip(x1, x2, sample_names, sample_categories[0]):
            ax1.text(x, y, name, ha='center', va="center",
                     color=plt.cm.spectral(cluster_label / 10.),
                     fontdict={'family': 'Arial', 'size': 10})

    # control aesthetics:
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    sns.plt.tight_layout()
    sns.plt.savefig(outputfile, bbox_inches=0)
    plt.clf()