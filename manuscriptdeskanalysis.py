"""
This file imports and performs the analysis in Pystyl for the Manuscript Desk project. See https://manuscriptdesk.uantwerpen.be
"""

import sys
import os
import ast
import traceback

from pystyl.corpus import Corpus
from pystyl.analysis import pca, tsne, distance_matrix, hierarchical_clustering, vnc_clustering, bootstrapped_distance_matrices, bootstrap_consensus_tree
from pystyl.manuscriptdeskvisualization import scatterplot, scatterplot_3d, clustermap, scipy_dendrogram, ete_dendrogram, bct_dendrogram

#import the data
try:
    full_textfilepath = ast.literal_eval(sys.argv[1])
    data = open(full_textfilepath)
    data = data.read()
    data = ast.literal_eval(data)

    removenonalpha = data['removenonalpha']
    lowercase = data['lowercase']
    tokenizer = data['tokenizer']
    minimumsize = data['minimumsize']
    maximumsize = data['maximumsize']
    segmentsize = data['segmentsize']
    stepsize = data['stepsize']
    removepronouns = data['removepronouns']
    vectorspace = data['vectorspace']
    featuretype = data['featuretype'] #not available yet
    ngramsize = data['ngramsize']
    mfi = data['mfi']
    minimumdf = data['minimumdf']
    maximumdf = data['maximumdf']

    base_outputpath = data['base_outputpath']
    full_outputpath1 = data['full_outputpath1']
    full_outputpath2 = data['full_outputpath2']

    visualization1 = data['visualization1']
    visualization2 = data['visualization2']

    texts_information_dict = data['texts']

except:
    sys.exit(1)

try:

    full_outputpath1  = full_outputpath1.replace("\\", "")
    full_outputpath2 = full_outputpath2.replace("\\", "")
    base_outputpath  = base_outputpath.replace("\\", "")
 
    #error if destination files already exist
    if os.path.isfile(full_outputpath1) or os.path.isfile(full_outputpath2):
        sys.exit(1)

    #make the base directory if it does not already exist
    if not os.path.exists(base_outputpath):
        os.makedirs(base_outputpath)

    #do the analysis and save the output
    corpus = Corpus(language='en')
    corpus.add_texts_manuscriptdesk(texts_information_dict = texts_information_dict)
    corpus.preprocess(alpha_only=removenonalpha, lowercase=lowercase)

    corpus.tokenize(min_size=minimumsize, max_size=maximumsize, tokenizer_option=tokenizer) #defaults can be used
    corpus.segment(segment_size=segmentsize, step_size = stepsize)#allow segmentatation? May cause problems..
    corpus.remove_tokens(rm_tokens=None, rm_pronouns=removepronouns, language='en') # watch out: if you do this before segment(), if will influence segment legths... (I would do it afterwards...)   Does removepronouns work!?

    corpus.vectorize(mfi=mfi, ngram_type='word', ngram_size=ngramsize, vector_space=vectorspace,min_df=minimumdf, max_df=maximumdf)

    def constructAndSaveVisualization(visualization, full_outputpath):

        """
        render a visualization of the analysis based on selection, and store this as an image
        :param visualization1: the visualization selected by the user
        :param full_outputpath: the full outputpath where the image will be stored
        """

        if visualization == 'dendrogram':
            dms = bootstrapped_distance_matrices(corpus, n_iter=100, random_prop=0.20, metric='manhattan')
            trees = [hierarchical_clustering(dm, linkage='ward') for dm in dms]
            bct = bootstrap_consensus_tree(corpus=corpus, trees=trees, consensus_level=0.5)
            bct_dendrogram(corpus=corpus, tree=bct, fontsize=8, color_leafs=False,mode='c', outputfile=full_outputpath, save=True)
        elif visualization == 'pcascatterplot':
            pca_coor, pca_loadings = pca(corpus)
            scatterplot(corpus, coor=pca_coor, loadings=pca_loadings, plot_type='static', outputfile=full_outputpath, save=True)
        elif visualization == 'tnsescatterplot':
            tsne_coor = tsne(corpus, nb_dimensions=2)
            scatterplot(corpus, coor=tsne_coor, nb_clusters=0, plot_type='static', outputfile=full_outputpath, save=True)
        elif visualization == 'distancematrix':
            dm = distance_matrix(corpus, metric='minmax')
            clustermap(corpus, distance_matrix=dm, fontsize=8, color_leafs=True, outputfile=full_outputpath, save=True)
        elif visualization == 'neighbourclustering':
            dm = distance_matrix(corpus, metric='minmax')
            vnc_tree = vnc_clustering(dm, linkage='ward')
            scipy_dendrogram(corpus, tree=vnc_tree, fontsize=8, color_leafs=False, outputfile=full_outputpath, save=True)
        return

    constructAndSaveVisualization(visualization1, full_outputpath1)
    constructAndSaveVisualization(visualization2, full_outputpath2)

except:
    sys.exit(1)
