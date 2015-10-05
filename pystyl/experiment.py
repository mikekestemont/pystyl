from . corpus import Corpus
from . analysis import *
from . visualization import *

class Experiment:
    """
    Represents a "standard" stylometric experiment,
    with a workflow similar to that in Stylometry with R.
    """

    def __init__(self, mode='GUI'):
        """
        Lightweight constructor
        """
        self.corpus = Corpus()
        self.mode = mode

    def import_data(self, directory, alpha_only, lowercase, extension):
        self.corpus.add_directory(directory=directory,
                                  ext=extension)
        self.corpus.preprocess(alpha_only=alpha_only,
                               lowercase=lowercase)

    def preprocess(self, segment_size, step_size,
                   min_size, max_size, tokenizer_option,
                   rm_tokens, rm_pronouns, language):
        self.corpus.tokenize(min_size=min_size,
                             max_size=max_size,
                             tokenizer_option=tokenizer_option)
        self.corpus.segment(segment_size=segment_size,
                            step_size=step_size)
        self.corpus.remove_tokens(rm_pronouns=rm_pronouns,
                                  language=language)

    def extract_features(self, mfi, ngram_type,
                         ngram_size, vector_space,
                         min_df, max_df):
        self.corpus.vectorize(mfi=mfi,
                         ngram_type=ngram_type,
                         ngram_size=ngram_size,
                         vector_space=vector_space)

    def visualize(self, outputfile=None, viz_type='pca', metric='minmax'):
        save, return_svg = False, False
        if self.mode == 'CMD_LINE':
            save = True
        if self.mode == 'GUI':
            return_svg = True

        if viz_type == 'pca':
            pca_coor, pca_loadings = pca(self.corpus)
            return scatterplot(self.corpus,
                    coor=pca_coor,
                    loadings=pca_loadings,
                    outputfile=outputfile,
                    save=save,
                    return_svg=return_svg)
        elif viz_type == 'pca_3d':
            pca_coor, pca_loadings = pca(self.corpus,
                                          nb_dimensions=3)
            pca_matrix_3d, _ = pca(self.corpus, nb_dimensions=3)
            return scatterplot_3d(self.corpus,
                    coor=pca_coor,
                    outputfile=outputfile,
                    save=save,
                    return_svg=return_svg)
        elif viz_type == 'clustermap':
            dm = distance_matrix(self.corpus,
                                 metric=metric)
            return clustermap(self.corpus,
                    distance_matrix=dm,
                    fontsize=8,
                    outputfile=outputfile,
                    save=save,
                    return_svg=return_svg)
        elif viz_type == 'dendrogram':
            dm = distance_matrix(self.corpus,
                    metric=metric)
            cluster_tree = hierarchical_clustering(dm,
                    linkage='ward')
            ete_dendrogram(corpus=self.corpus,
                    tree=cluster_tree,
                    fontsize=8,
                    mode='c',
                    outputfile=outputfile,
                    save=save,
                    return_svg=return_svg)





