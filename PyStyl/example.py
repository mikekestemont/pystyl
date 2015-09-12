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
from corpus import Corpus
from vectorization import Vectorizer
from analysis import pca, tsne
from visualization import scatterplot_2d, scatterplot_3d

c = Corpus()
#c.add_texts_from_directory(directory='../data/dummy1')
c.add_texts_from_directory(directory='../data/dummy2')
c.segment(segment_size=5000, step_size=None, min_size=1000, max_size=20000)
v = Vectorizer(mfi=500, ngram_type='word',
               ngram_size=1, vector_space='tf_std',
               lowercase=True, vocabulary=None)

X = v.fit_transform(c.texts)
X = v.remove_pronouns(X, language='en')
reduced_X, loadings = pca(X, nb_dimensions=2)
scatterplot_2d(reduced_X, sample_names=c.titles, nb_clusters=0, loadings=loadings,
                feature_names=v.features, sample_categories=(c.target_ints, c.target_idx))

reduced_X, loadings = pca(X, nb_dimensions=3)
scatterplot_3d(reduced_X, sample_names=c.titles, nb_clusters=0, sample_categories=(c.target_ints, c.target_idx))

#tsne(X=X, sample_names=c.titles, sample_categories=(c.target_ints, c.target_idx))

#### 3D barplots for word frequencies!

#for feature in feature_type:
#    e.add_feature((ngram_type, ngram_range, nb_mfi, culling_rate, remove_pronouns, lowercase) # stacking features in self.X

#e = e.load('settings.txt')

#e.load_corpus(dir="corpus")
"""
e.set_language(lang='nl')

e.visualize(method='pca', outfilename='vvv.png', label_fontsize, labels_to_include)

e.save('settings.txt')
"""
