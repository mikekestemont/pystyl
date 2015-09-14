from __future__ import print_function

import os
import codecs
import glob
from operator import itemgetter

from nltk.tokenize import RegexpTokenizer

def untokenize(words):
    return ' '.join(words)

class Corpus:

    def __init__(self, texts=[], titles=[], target_ints=[], target_idx={}):
        self.texts = texts
        self.titles = titles
        self.target_ints = target_ints # integers corresponding to category names
        self.target_idx = target_idx # actual category names as strings

    def add_texts_from_directory(self, directory, encoding='utf-8', ext='.txt'):
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

    def temporal_sort(self):
        # check whether the categories are properly formatted:
        for cat in self.target_idx: # no generator here, to allow detailed error message
            if not cat.isdigit():
                raise ValueError('Category label is not a number: %s' %(cat))

        zipped = [(int(self.target_idx[target_int]), text, title, target_int) \
                       for text, title, target_int in zip(self.texts, self.titles, self.target_ints)]
        zipped.sort(key=itemgetter(0), reverse=False)

        _, self.texts, self.titles, self.target_ints = zip(*zipped)


    def segment(self, segment_size=0, step_size=0, min_size=0, max_size=0):
        """
        Important: the tokenizer will have a great influence on the segmentation procedure!
        """
        # sanity checks:
        if max_size < min_size:
            raise ValueError('Segmentation parameter error: max_size < min_size')
        if segment_size > max_size:
            raise ValueError('Segmentation parameter error: segment_size > max_size')
        if step_size > segment_size:
            raise ValueError('Segmentation parameter error: step_size > segment_size')
        if not self.texts:
            raise ValueError('No texts loaded yet.')

        tokenizer = RegexpTokenizer(r'\w+')

        tokenized_texts = []
        for i, text in enumerate(self.texts):
            tokens = tokenizer.tokenize(text)
            if max_size:
                tokens = tokens[:max_size] # cut
            if min_size and len(tokens) < min_size:
                print("Title: %s only has %d tokens (< sample_size = %d) -> ignored" % (title, len(text)))
                continue
            tokenized_texts.append(tokens)

        # segment:
        if not segment_size:
            self.texts = [untokenize(t) for t in tokenized_texts]
        else:
            if not step_size:
                step_size = segment_size

            texts, titles, target_ints = [], [], []

            for text, title, target_int in zip(tokenized_texts, self.titles, self.target_ints):
                start_idx, end_idx, sample_cnt = 0, segment_size, 1
                while end_idx <= len(text):
                    texts.append(text[start_idx: end_idx])
                    titles.append(title + "_" + str(sample_cnt))
                    target_ints.append(target_int)
                    # update:
                    sample_cnt += 1
                    start_idx += step_size
                    end_idx += step_size

            self.texts = [untokenize(t) for t in texts]
            self.titles, self.target_ints = titles, target_ints

    def __len__(self):
        return len(self.texts)

    def __repr__(self):
        return "<Corpus(%s texts)> " % len(self)

    def __str__(self):
        info_string = "Corpus contains:\n"
        for text, title, target_int in zip(self.texts, self.titles, self.target_ints):
            info_string += "\t- %s\t(cat: %s):\t%r[...]\n" % (title, self.target_idx[target_int], ' '.join(text[:7]))
        return info_string

            
