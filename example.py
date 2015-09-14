"""
params:
- visualize / classify
- which tokenizer
- distance metric: euclidean/cosine, manhattan, minmax, 
- method for visualize: pca or hac or vnc or tsne
- names fontsizes, what to include: author, genre, idx etc.
- outfilename
- settingsfilename
- select files manually

# differences:
- we only except utf8
- no xml parsing
- not bootstrap looping yet
- no mds, no bootstrap
- only py3
"""


#from experiment import Experiment
from PyStyl.corpus import Corpus
from PyStyl.vectorization import Vectorizer
from PyStyl.analysis import pca, tsne, distance_matrix, hierarchical_clustering
from PyStyl.visualization import scatterplot_2d, scatterplot_3d, clustermap, dendrogram

c = Corpus()
#c.add_texts_from_directory(directory='../data/dummy1')
c.add_texts_from_directory(directory='data/dummy2')
c.segment(segment_size=10000, min_size=1000, max_size=20000)
v = Vectorizer(mfi=500, ngram_type='word',
               ngram_size=1, vector_space='tf',
               lowercase=True, vocabulary=None)

X = v.fit_transform(c.texts)
X = v.remove_pronouns(X, language='en') # refit?

"""
# pca in 2d:
reduced_X, loadings = pca(X, nb_dimensions=2)
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
"""

dm = distance_matrix(X, 'minmax')
clustermap(dm, sample_names=c.titles, sample_categories=(c.target_ints, c.target_idx), fontsize=8)
cluster_tree = hierarchical_clustering(dm, linkage='ward')
dendrogram(cluster_tree, sample_names=c.titles, sample_categories=(c.target_ints, c.target_idx), fontsize=8)


#for feature in feature_type:
#    e.add_feature((ngram_type, ngram_range, nb_mfi, culling_rate, remove_pronouns, lowercase) # stacking features in self.X

#e = e.load('settings.txt')

#e.save('settings.txt')
