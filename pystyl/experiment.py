from . corpus import Corpus
from . analysis import pca, tsne, distance_matrix, hierarchical_clustering, vnc_clustering, bootstrapped_distance_matrices, bootstrap_consensus_tree
from . visualization import scatterplot, scatterplot_3d, clustermap, scipy_dendrogram, ete_dendrogram, bct_dendrogram

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

    def import_data(self, directory, alpha_only, lowercase):
        self.corpus.add_directory(directory=directory)
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

    def visualize(self, outputfile):
        pca_coor, pca_loadings = pca(self.corpus)
        save, return_svg = False, False
        if self.mode == 'CMD_LINE':
            save = True
        if self.mode == 'GUI':
            return_svg = True
        scatterplot(self.corpus,
                    coor=pca_coor,
                    loadings=pca_loadings,
                    outputfile=outputfile,
                    save=save,
                    return_svg=return_svg)




