from __future__ import print_function

import os
import codecs
import glob
from operator import itemgetter
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
elif sys.version_info[0] == 3:
    import pickle

from nltk.tokenize import WhitespaceTokenizer, RegexpTokenizer

from pystyl.vectorization import Vectorizer

std_output_path = os.path.dirname(os.path.abspath(__file__))+'/../output/'

def get_tokenizer(option=None):
    """
    Notes:
    * the tokenizer is a property of the corpus, instead of the vectorizer
      in PyStyl, because the tokenizer is crucial to the segmentation.
    * right now, the tokenizers aren't a property of the Corpus objects,
      because they cannot pickled.
    """
    if option == None or option == 'whitespace':
        return WhitespaceTokenizer()
    elif option == 'words':
        return RegexpTokenizer(r'\w+')
    else:
        raise ValueError('Invalid tokenization option: %s' %(option))

class Corpus:

    def __init__(self, texts=[], titles=[], target_ints=[],
                       target_idx={}, language=None, tokenized_texts=None):
        self.language = language
        self.texts = texts
        self.tokenized_texts = tokenized_texts
        self.titles = titles
        self.target_ints = target_ints # integers corresponding to category names
        self.target_idx = target_idx # actual category names as strings
        self.tokenizer_option = None
        self.vectorizer = None

    def add_directory(self, directory, encoding='utf-8', ext='.txt'):
        """
        - add texts under directory to the corpus
        - by default, assumes utf8 in files and naming convention: <category>_<title>.txt 
        """

        directory = os.path.abspath(directory)
        if not os.path.isdir(directory):
            raise IOError('Folder %s does not exist...' %(directory))

        print('Adding texts from:', directory)
        if not self.target_idx:
            self.target_idx = []
            self.texts, self.titles, self.target_ints = [], [], []

        for filename in sorted(glob.glob(directory+'/*'+ext)):
            if filename.startswith("."):
                continue

            if '_' not in filename or not filename.endswith('.txt') or filename.count('_') > 1:
                raise ValueError("Filename: "+filename+" wrongly formatted (should be: category_title.txt)")

            with codecs.open(filename, mode='r', encoding=encoding) as infile:
                text = infile.read()
                if text.strip():
                    target_name, title = os.path.basename(filename).replace(ext, '').split('_')

                    if target_name not in self.target_idx:
                        self.target_idx.append(target_name)
                    
                    target_int = self.target_idx.index(target_name)
                    
                    self.texts.append(text)
                    self.titles.append(title)
                    self.target_ints.append(target_int)

                else:
                    print("Ignored: "+filename+" (does not contain any text...)")

    def preprocess(self, alpha_only=True, lowercase=True):
        for idx, text in enumerate(self.texts):
            if lowercase:
                text = text.lower()
            if alpha_only:
                text = ''.join([c for c in text if c.isalpha() or c.isspace()])
            self.texts[idx] = text

    def tokenize(self, min_size=0, max_size=0, tokenizer_option=None):
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
        if not self.tokenized_texts:
            raise ValueError('Texts not tokenized yet.')

        if rm_pronouns:
            if language:
                # warning: this will change the language setting corpus-wide
                self.language = language
            if not self.language:
                raise ValueError('No language set for corpus (cf. pronouns)')
            if self.language not in ('en'):
                raise ValueError('No pronouns available for: %s' %(self.language))
            pronoun_path = os.path.dirname(os.path.abspath(__file__))+'/pronouns/'
            pronouns = {w.strip() for w in open(pronoun_path+self.language+'.txt') if not w.startswith('#')}
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
        Important: the tokenizer will have a great influence on the segmentation procedure!
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
                 max_df=1.0, min_df=0.0 ù):
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
        return [' '.join(t) for t in self.tokenized_texts]

    def __len__(self):
        if self.tokenized_texts:
            return len(self.tokenized_texts)
        elif self.texts:
            return len(self.texts)
        else:
            return 0

    def __repr__(self):
        return "<Corpus(%s texts)> " % len(self)

    def __str__(self):
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

            
