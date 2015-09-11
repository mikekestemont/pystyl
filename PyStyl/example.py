"""
params:
- input directory, with title(_author(_genre(_date))).txt
- segment_size (tokens)
- minimum text size (tokens) / max text size
- visualize / classify
- nb_mfi
- which tokenizer
- vector space model: tf, tfidf, binary, std
- distance metric: euclidean/cosine, manhattan, minmax, 
- scale: bool
- to unit vector
- ngram_type, ngram_size, char_ngrams_across_words
- culling rate
- method for visualize: pca or hac or vnc or tsne
- names fontsizes, what to include: author, genre, idx etc.
- outfilename
- settingsfilename
- remove pronouns
- lowercase or not
- language
- select files manually

# differences:
- we only except utf8
- no xml parsing
- not bootstrap looping yet
- no mds, no bootstrap
- only py3
"""


from experiment import Experiment
from corpus import Corpus

c = Corpus()
c.add_texts_from_directory(directory='../data/dummy1')
c.add_texts_from_directory(directory='../data/dummy2')
c.segment(segment_size=5000, step_size=1000, min_size=1000, max_size=20000)
print(c)

#for feature in feature_type:
#    e.add_feature((ngram_type, ngram_range, nb_mfi, culling_rate, remove_pronouns, lowercase) # stacking features in self.X

#e = e.load('settings.txt')

#e.load_corpus(dir="corpus")
"""
e.set_language(lang='nl')

e.visualize(method='pca', outfilename='vvv.png', label_fontsize, labels_to_include)

e.save('settings.txt')
"""
