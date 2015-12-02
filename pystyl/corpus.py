# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import codecs
import glob
from operator import itemgetter
from pkg_resources import resource_string
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
elif sys.version_info[0] == 3:
    import pickle

from nltk.tokenize import WhitespaceTokenizer, RegexpTokenizer

from . vectorization import Vectorizer


def get_tokenizer(option=None):
    """
    Return a nltk tokenizer.

    Parameters
    ----------
    option : string, default=None
        Which tokenizer to load currently supports:
        - 'whitespace' > nltk's `WhitespaceTokenizer`
        - 'words' > nltk's `RegexpTokenizer(r'\w+')`

    Returns
    ----------
    An nltk tokenizer object.

    """
    if option == None or option == 'whitespace':
        return WhitespaceTokenizer()
    elif option == 'words':
        return RegexpTokenizer(r'\w+')
    else:
        raise ValueError('Invalid tokenization option: %s' %(option))

class Corpus:

    def __init__(self, texts=[], titles=[], target_ints=[],
                       target_idx=[], language=None, tokenized_texts=None):
        """
        A class to represent corpora or a collection of texts. Note
        that titles should be unique.

        """
        self.language = language
        self.texts = texts
        self.tokenized_texts = tokenized_texts
        self.titles = titles # should be unique
        self.target_ints = target_ints # integers corresponding to category names
        self.target_idx = target_idx # actual category names as strings
        self.tokenizer_option = None
        self.vectorizer = None

    def add_directory(self, directory, encoding='utf-8', ext='txt'):
        """
        Add the texts under a directory to the corpus. Consecutive calls
        will add new texts to the corpus, instead of overwriting the old ones.

        Parameters
        ----------
        directory : string
            The path to the directory.
            Note that texts under the directory have to conform to the 
            following syntax: <category>_<title>.<ext>

        encoding : string, default='utf-8'
            The encoding of your files.

        ext : str, default='txt'
            The extension of the text files under directory
            Only filenames with the extension will be loaded

        """

        directory = os.path.abspath(directory)
        if not os.path.isdir(directory):
            raise IOError('Folder %s does not exist...' %(directory))

        if not self.target_idx:
            self.target_idx = []
            self.texts, self.titles, self.target_ints = [], [], []

        for filename in sorted(glob.glob(directory+'/*.'+ext)):

            basename = os.path.basename(filename)
            if basename.startswith('.'):
                continue

            if '_' not in basename or not basename.endswith('.'+ext) or basename.count('_') > 1:
                raise ValueError('Filename: '+basename+' wrongly formatted (should be: category_title.ext)')

            with codecs.open(filename, mode='r', encoding=encoding) as infile:
                text = infile.read()
                if text.strip():
                    target_name, title = os.path.basename(filename).replace('.'+ext, '').split('_')
                    self.add_text(text=text, title=title, target_name=target_name)

                else:
                    print("Ignored: "+filename+" (does not contain any text...)")

    def add_text(self, text, title, target_name):
        """
        Add a single new text to the corpus.

        Parameters
        ----------
        text : str
            The text to be added.

        title : str
            Title of the text to be added.

        target_name : str
            Category of the text to be added.

        """
        if title in self.titles:
            raise ValueError('Titles should be unique: %s is already in the corpus' %(title))
        if target_name not in self.target_idx:
            self.target_idx.append(target_name)
                    
        target_int = self.target_idx.index(target_name)
                    
        self.texts.append(text)
        self.titles.append(title)
        self.target_ints.append(target_int)

    def add_texts(self, texts, titles, target_names):
        """
        Add series of new texts to the corpus.

        Parameters
        ----------
        texts : list-like
            The text to be added.

        title : list-like
            Titles of the texts to be added.

        target_name : list-like
            Categories of the texts to be added.

        """
        if not texts or not titles or not target_names:
            raise ValueError('Please specify a text,\
                              title and target_name\
                              for each text')
        if not len(texts) == len(titles)\
            and len(titles) == len(target_names):
            raise ValueError('Nb of texts, titles and\
                             target_names should be the\
                             same')
        for text, title, target_name in zip(texts, titles, target_names):
            self.add_text(text=text, title=title, target_name=target_name)


    def preprocess(self, alpha_only=True, lowercase=True):
        """
        Preprocess the (untokenized) texts in the corpus.

        Parameters
        ----------
        alpha_only : boolean, default=True
            Whether or not only to keep alphabetic symbols.
            Whitespace characters remain unaffected.

        lowercase : boolean, default=True
            Whether or not to lowercase all characters.

        """
        for idx, text in enumerate(self.texts):
            if lowercase:
                text = text.lower()
            if alpha_only:
                text = ''.join([c for c in text if c.isalpha() or c.isspace()])
            self.texts[idx] = text

    def tokenize(self, min_size=0, max_size=0, tokenizer_option=None):
        """
        Tokenize the texts in the corpus and normalize text length.

        Parameters
        ----------
        min_size : int, default=0
            Minimum size of texts (in tokens), to be included
            in the set of tokenized texts.
            An error will be raised if `min_size` > `max_size`
            If `min_size`=0, no texts will be left out.

        max_size : int, default=0
            Maximum size of texts (in tokens). Longer texts
            will be truncated to max_size after tokenization.
            An error will be raised if `max_size` > `min_size`
            If `max_size`=0, no etxs will truncated.

        tokenizer_option : str, default=None
            Select the `nltk` tokenizer to be used. Currently
            supports: 'whitespace' (split on whitespace)
            and 'words' (alphabetic series of characters).

        """
        # sanity check:
        if min_size and max_size and min_size > max_size:
            raise ValueError('Tokenization error: min_size > max_size')

        if not self.texts:
            raise ValueError('No texts added to the corpus yet')

        self.min_size = min_size
        self.max_size = max_size

        if tokenizer_option:
            self.tokenizer_option = tokenizer_option
        tokenizer = get_tokenizer(option=self.tokenizer_option)

        self.tokenized_texts = []
        for i, text in enumerate(self.texts):
            tokens = tokenizer.tokenize(text)
            if self.max_size:
                tokens = tokens[:self.max_size] # cut
            if self.min_size and len(tokens) < self.min_size:
                print("Title: %s only has %d tokens (< sample_size = %d) -> ignored" % (self.titles[i], len(text)))
                continue
            self.tokenized_texts.append(tokens)

    def remove_tokens(self, rm_tokens=[], rm_pronouns=False, language=None):
        """
        Remove specific tokens from the tokenized texts.
        Must be called before `corpus.vectorize()` to have effect.

        Parameters
        ----------
        rm_tokens : list of str, default=[]
            List of tokens to be removed.
            Currently not sensitive to capitalization.

        rm_pronouns : boolean, default=False
            Whether to remove personal pronouns.
            If the `corpus.language` is supported,
            we will load the relevant list from 
            under `pystyl/pronouns`.
            The pronoun lists are identical to those
            for 'Stylometry with R'.

        language : str, default=None
            Option to (re)set the `language` property
            of the corpus.
            Currently supported:
                - 'en' (English)
                - 'nl' (Dutch)

        """
        if not self.tokenized_texts:
            raise ValueError('Texts not tokenized yet.')

        if rm_pronouns:
            if language:
                # warning: this will change the language setting corpus-wide
                self.language = language
            if not self.language:
                raise ValueError('No language set for corpus (cf. pronouns)')
            if self.language not in ('en', 'nl'):
                raise ValueError('No pronouns available for: %s' %(self.language))
            pronouns = {w.strip() for w in \
                            resource_string(__name__,
                                'pronouns/'+self.language+'.txt')\
                            if not w.startswith('#')}
        else:
            pronouns = set()

        if rm_tokens:
            rm_tokens = set([r.lower() for r in rm_tokens])
        else:
            rm_tokens = set()

        rm = rm_tokens.union(pronouns)
        for idx, text in enumerate(self.tokenized_texts):
            self.tokenized_texts[idx] = [w for w in text if w.lower() not in rm]


    def segment(self, segment_size=0, step_size=0):
        """
        Segment the tokenized_texts into smaller units.
        Subsequent calls will overwrite previous segmentations.
        Trailing tokens at the end of a text will be ignored
        if they cannot form an entire segment anymore.

        Parameters
        ----------
        segment_size : int, default=0
            The size of the segments to be extracted
            (in tokens).
            If `segment_size`=0, no segmentation will be
            applied to the tokenized texts.
            A error will be raised if
            `segment_size` > `self.max_size`

        step_size : int, default=0
            The nb of words in between two consecutive
            segments (in tokens).
            If `step_size`=zero, non-overlapping segments
            will be created. Else, segments will partially
            overlap.
            A error will be raised if `step_size` > `segment_size`
            or `step_size` > `self.max_size`

        """

        # sanity checks:
        if self.max_size and segment_size > self.max_size:
            raise ValueError('Segmentation error: segment_size > self.max_size')
        if step_size and segment_size and step_size > segment_size:
            raise ValueError('Segmentation error: step_size > segment_size')
        if not self.tokenized_texts:
            raise ValueError('Texts not tokenized yet.')

        self.step_size = step_size
        self.segment_size = segment_size

        # segment if necessary (else we leave the tokenized_texts unchanged):
        if self.segment_size:
            if not self.step_size:
                self.step_size = self.segment_size

            tmp_texts, tmp_titles, tmp_target_ints = [], [], []

            for text, title, target_int in zip(self.tokenized_texts, self.titles, self.target_ints):
                start_idx, end_idx, sample_cnt = 0, self.segment_size, 1
                while end_idx <= len(text):
                    tmp_texts.append(text[start_idx : end_idx])
                    tmp_titles.append(title + "_" + str(sample_cnt))
                    tmp_target_ints.append(target_int)
                    # update:
                    sample_cnt += 1
                    start_idx += self.step_size
                    end_idx += self.step_size

            self.tokenized_texts, self.titles, self.target_ints = \
                tmp_texts, tmp_titles, tmp_target_ints

    def temporal_sort(self):
        """
        Function which will garantee that `tokenized_texts` are
        sorted in the correct order, e.g. in terms of chronology
        or order of appearance in a text.
        This function assumes that all the texts' category label is
        an integer, reflecting the correct order of the original
        texts. Else, an error will be raised.

        """
        # check whether the categories are properly formatted:
        for cat in self.target_idx: # no generator here, to allow detailed error message
            if not cat.isdigit():
                raise ValueError('Corpus cannot be sorted: category label is not a number: %s' %(cat))
        if not self.tokenized_texts:
            raise ValueError('Please only sort the corpus after tokenizing it')
        if self.vectorizer:
            raise ValueError('You cannot sort the corpus after it has been vectorized')

        zipped = [(int(self.target_idx[target_int]), text, title, target_int) \
                       for text, title, target_int in \
                            zip(self.tokenized_texts, self.titles, self.target_ints)]
        zipped.sort(key=itemgetter(0), reverse=False)

        _, self.tokenized_texts, self.titles, self.target_ints = zip(*zipped)

    def vectorize(self, mfi=500, ngram_type='word', ngram_size=1,
                 vector_space='tf', vocabulary=None,
                 max_df=1.0, min_df=0.0):
        """
        Function to vectorize a corpus. Will add a vectorizer to corpus,
        overwriting previous calls to this method. For the parametrization,
        see the docs for `vectorization.Vectorize()` to which all args are passed.
        Will raise an error if `tokenized_texts` are unavailable in the corpus.

        Returns
        -------
        feature_names : list
            The names of the final features extracted by the vectorizer.

        """
        if not self.tokenized_texts:
            print('Warning: corpus has not been tokenized yet: running tokenization with default settings first')
            self.tokenize()

        self.vectorizer = Vectorizer(mfi=mfi,
                                     ngram_type=ngram_type,
                                     ngram_size=ngram_size,
                                     vector_space=vector_space,
                                     vocabulary=vocabulary,
                                     min_df=min_df,
                                     max_df=max_df)

        if ngram_type == 'word':
            self.vectorizer.vectorize(self.tokenized_texts)
        elif ngram_type in ('char', 'char_wb'):
            self.vectorizer.vectorize(self.get_untokenized_texts())
        else:
            raise ValueError('Unsupported feature type: %s' %(ngram_type))
        return self.vectorizer.feature_names

    def get_untokenized_texts(self):
        """
        Get the tokenized texts in an untokenized version: 
        all token are re-joined using a single space character.
        This is useful for the extraction of characters ngram_size
        by the vectorizer.

        Returns
        -------
        untokenized_texts : list
            The names of the final features extracted by the vectorizer.

        """
        return [' '.join(t) for t in self.tokenized_texts]

    def __len__(self):
        """
        Count the nb of texts currently in corpus.

        Returns
        -------
        nb_texts : int
            If `tokenized_texts` is available, the nb
            of tokenized texts will be returned.
            Else, the nb of untokenized texts is returned.

        """
        if self.tokenized_texts:
            return len(self.tokenized_texts)
        elif self.texts:
            return len(self.texts)
        else:
            return 0

    def __repr__(self):
        """
        Returns a simple string representation of the corpus,
        indicating the nb of texts currently available.

        """
        return "<Corpus(%s texts)> " % len(self)

    def __str__(self):
        """
        Returns a string representation of the corpus,
        indicating the nb of texts currently available
        and the title and category for each text, as well
        the first 10 tokens of each text, if tokenized texts
        are available, and else the 30 first characters of
        each text.

        """
        info_string = repr(self)
        if self.tokenized_texts:
            info_string += '\nTokenized texts:'
            for text, title, target_int in zip(self.tokenized_texts, self.titles, self.target_ints):
                info_string += "\n\t- %s\t(cat: %s):\t%r[...]" % (title, self.target_idx[target_int], ' '.join(text[:10]))
        elif self.texts:
            info_string += '\nUntokenized texts:\n'
            for text, title, target_int in zip(self.texts, self.titles, self.target_ints):
                info_string += "\n\t- %s\t(cat: %s):\t%r[...]" % (title, self.target_idx[target_int], ''.join(text[:30]))
        else:
            info_string += 'No texts in corpus.'
        return info_string

            
