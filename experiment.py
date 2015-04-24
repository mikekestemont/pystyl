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
        self.method = "pca"#"cluster"

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
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            Xs = StandardScaler().fit_transform(self.X.toarray())
            P = PCA(n_components=2)
            Xr = P.fit_transform(Xs)
            loadings = P.components_.transpose()
            #sb.set_style("darkgrid")
            fig, ax1 = plt.subplots()
            # first samples:    
            x1, x2 = Xr[:,0], Xr[:,1]
            ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none');
            for x,y,l in zip(x1, x2, self.corpus.titles):
                ax1.text(x, y, l ,ha='center', va="center", size=10, color="darkgrey")
            # now loadings:
            #sb.set_style("dark")
            ax2 = ax1.twinx().twiny()
            l1, l2 = loadings[:,0], loadings[:,1]
            ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none');
            for x,y,l in zip(l1, l2, self.vectorizer.get_feature_names()):
                ax2.text(x, y, l ,ha='center', va="center", size=10, color="black")
            plt.show()

    def analysis(self):
        pass
        

exp = Experiment()
exp.preprocess()
print(exp.corpus)
exp.feature_extraction()




