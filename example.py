import os

#from experiment import Experiment
from PyStyl.corpus import Corpus
from PyStyl.vectorization import Vectorizer
from PyStyl.analysis import pca, tsne, distance_matrix, hierarchical_clustering, vnc_clustering
from PyStyl.visualization import scatterplot_2d, scatterplot_3d, scatterplot_2d_bokeh
from PyStyl.visualization import clustermap, scipy_dendrogram, ete_dendrogram

if not os.path.isdir('../output/'):
    os.mkdir('../output/')

c = Corpus()
#c.add_texts_from_directory(directory='data/dummy1')
c.add_texts_from_directory(directory='data/sorted')
c.temporal_sort() # we assume that the categpries are sortable integers, indicating some order (e.g. date of composition)
c.segment(segment_size=3000, min_size=1000, max_size=20000)

c.save()
c = Corpus.load()

v = Vectorizer(mfi=500, ngram_type='word',
               ngram_size=1, vector_space='tf_std',
               lowercase=True, vocabulary=None)

X = v.fit_transform(c.texts)
X = v.remove_pronouns(X, language='en')

v.save()
v = Vectorizer.load()


reduced_X, loadings = pca(X, nb_dimensions=2)

# pca in 2d:
scatterplot_2d_bokeh(reduced_X, sample_names=c.titles, nb_clusters=4, sample_categories=(c.target_ints, c.target_idx))

scatterplot_2d(reduced_X, sample_names=c.titles, nb_clusters=4, loadings=False,
                feature_names=v.features, sample_categories=(c.target_ints, c.target_idx))
# pca in 3d:
reduced_X, loadings = pca(X, nb_dimensions=3)
scatterplot_3d(reduced_X, sample_names=c.titles, nb_clusters=4, sample_categories=(c.target_ints, c.target_idx))

# tsne in 2d:
reduced_X = tsne(pca(X, nb_dimensions=10)[0], nb_dimensions=2)
scatterplot_2d(reduced_X, sample_names=c.titles, nb_clusters=4, loadings=False,
                feature_names=v.features, sample_categories=(c.target_ints, c.target_idx))
# tsne in 3d:
reduced_X = tsne(pca(X, nb_dimensions=10)[0], nb_dimensions=3)
scatterplot_3d(reduced_X, sample_names=c.titles, nb_clusters=4, sample_categories=(c.target_ints, c.target_idx))


dm = distance_matrix(X, 'minmax')

clustermap(dm, sample_names=c.titles, sample_categories=(c.target_ints, c.target_idx), fontsize=8)
cluster_tree = hierarchical_clustering(dm, linkage='ward')
scipy_dendrogram(cluster_tree, sample_names=c.titles, sample_categories=(c.target_ints, c.target_idx), fontsize=8)
ete_dendrogram(cluster_tree, sample_names=c.titles, sample_categories=(c.target_ints, c.target_idx), fontsize=8)


vnc_tree = vnc_clustering(dm, linkage='ward') # still need to work the sorting...
scipy_dendrogram(vnc_tree, sample_names=c.titles, sample_categories=(c.target_ints, c.target_idx), fontsize=8)

