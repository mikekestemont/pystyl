from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import utilities
import matplotlib
from scipy.cluster.hierarchy import ward, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from corpus import *
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA as sklearnPCA

class Experiment:
    def __init__(self):
        # set all parameters
        self.data_home = "data/dummy"
        self.tokenization_option = ""
        self.sampling_method = "slicing"
        self.sample_size = 10000
        self.step_size = 10000
        self.culling_rate = 0.0
        self.MFI = 100
        self.ngram_type = "word" # sklearn's analyser: word', 'char', 'char_wb'
        self.ngram_size = 1 # we only support one ngram-type like stylo right now
        self.method = "cluster"

    def preprocess(self):
        self.corpus = Corpus.load(self.data_home)
        print(self.corpus) # calls __str__()
        self.corpus.tokenize(self.tokenization_option) # tokenize the corpus to prepare it for the sampling
        self.corpus = self.corpus.sample(method=self.sampling_method,
                                         sample_size=self.sample_size,
                                         step_size=self.step_size) # redefine the corpus

    def feature_extraction(self):
        print(self.corpus) # calls __str__()
        if self.ngram_type in ("char", "char_wb"):
            self.corpus.untokenize() # make strings again out of the text
            tokenizer = None # use sklearn char-tokenizer
        else:
            tokenizer = utilities.identity # if dealing with words, make sure sklearn doesn't start re-tokenizing etc.
        self.vectorizer = TfidfVectorizer(preprocessor=utilities.identity,
                                          analyzer=self.ngram_type,
                                          tokenizer=tokenizer,
                                          ngram_range=(self.ngram_size, self.ngram_size),
                                          stop_words=None,
                                          min_df=self.culling_rate,
                                          use_idf=False,
                                          max_features=self.MFI)
        self.X = self.vectorizer.fit(self.corpus.texts).transform(self.corpus.texts)
        # BCT loop can only start here... we sample from a pre-determined vocabulary
        print(sorted(self.vectorizer.vocabulary_))
        print self.corpus.target_names
        if self.method == "cluster":
            dist = euclidean_distances(self.X)
            linkage_matrix = ward(dist)
            authors = [self.corpus.target_names[t] for t in self.corpus.targets]
            labels = [author+"_"+title for author, title in zip(authors, self.corpus.titles)]
            colors = plt.rcParams['axes.color_cycle'][:len(authors)]
            dendrogram(linkage_matrix, orientation="left", labels=labels)
            ax = plt.gca()
            ax_labels = ax.get_ymajorticklabels() # this has to be xmajorticklabels() if you change the orientation of the dendrogram!
            for i in range(len(ax_labels)):
                target_name = ax_labels[i].get_text().split("_")[0]
                target_index = self.corpus.target_names.index(target_name)
                ax_labels[i].set_color(colors[target_index])
            plt.show()
        elif self.method == "pca":
            authors = [self.corpus.target_names[t] for t in self.corpus.targets]
            sklearn_pca = sklearnPCA(n_components=2)
            Xr = sklearn_pca.fit_transform(self.X.toarray())
            plt.figure()
            author_marks = list(plt.MarkerStyle.markers)[:len(self.corpus.targets)]
            for i in range(Xr.shape[0]):
                marker = author_marks[self.corpus.targets[i]]
                c1 = plt.scatter(Xr[i,0], Xr[i,1], color="r", marker=marker, label="Charlotte")
            plt.xlabel('1st PC')
            plt.ylabel('2nd PC')
            #plt.legend([c1, c2, c3], ["Char", "Anne", "Em"])
            plt.title('Principal Components Analysis')
            plt.show()

    def analysis(self):
        pass
        

exp = Experiment()
exp.preprocess()
print(exp.corpus)
exp.feature_extraction()




