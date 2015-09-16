import os

from pystyl.corpus import Corpus
from pystyl.analysis import pca, tsne, distance_matrix, hierarchical_clustering, vnc_clustering
from pystyl.visualization import scatterplot, scatterplot_3d, clustermap, scipy_dendrogram, ete_dendrogram

if not os.path.isdir('../output/'):
    os.mkdir('../output/')

corpus = Corpus(language='en')
corpus.add_directory(directory='data/dummy')
#corpus.add_directory(directory='data/sorted')
corpus.preprocess(alpha_only=True, lowercase=True)
corpus.tokenize()
corpus.segment(segment_size=20000, step_size=20000)
corpus.remove_tokens(rm_tokens=None, rm_pronouns=True, language='en') # watch out: if you do this before segment(), if will influence segment legths... (I would do it afterwards...)
#corpus.temporal_sort() # we assume that the categpries are sortable integers, indicating some order (e.g. date of composition)
print(corpus)
corpus.vectorize(mfi=100, ngram_type='word', ngram_size=1, vector_space='tf_std')
print(corpus.vectorizer.feature_names)


pca_coor, pca_loadings = pca(corpus, nb_dimensions=2)
scatterplot(corpus, coor=pca_coor, nb_clusters=0, loadings=pca_loadings, plot_type='static')
scatterplot(corpus, coor=pca_coor, nb_clusters=0, plot_type='interactive')
pca_matrix_3d, _ = pca(corpus, nb_dimensions=3)
scatterplot_3d(corpus, coor=pca_matrix_3d, nb_clusters=4)

tsne_coor = tsne(corpus, nb_dimensions=2)
scatterplot(corpus, coor=tsne_coor, nb_clusters=0, plot_type='static')
scatterplot(corpus, coor=tsne_coor, nb_clusters=0, plot_type='interactive')
tsne_coor_3d = tsne(corpus, nb_dimensions=3)
scatterplot_3d(corpus, coor=tsne_coor_3d, nb_clusters=4)

dm = distance_matrix(corpus, 'minmax')
clustermap(corpus, distance_matrix=dm, fontsize=8, color_leafs=True)

cluster_tree = hierarchical_clustering(dm, linkage='ward')
scipy_dendrogram(corpus=corpus, tree=cluster_tree, fontsize=8, color_leafs=False)
ete_dendrogram(corpus=corpus, tree=cluster_tree, fontsize=8, color_leafs=False, mode='c')

vnc_tree = vnc_clustering(dm, linkage='ward')
scipy_dendrogram(corpus, tree=vnc_tree, fontsize=8, color_leafs=False)
ete_dendrogram(corpus, tree=vnc_tree, fontsize=8, color_leafs=False, mode='r')


