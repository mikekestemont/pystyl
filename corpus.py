import os, codecs, utilities
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer # import this for tokenizer

class Corpus:
    def __init__(self, texts=[], titles=[], targets=[], target_names=[]):
        self.texts = texts
        self.titles = titles
        self.targets = targets # integers corresponding to category names
        self.target_names = target_names # actual category names as strings

    @classmethod
    def load(cls, data_home, encoding="utf-8"):
        # load data from data_home and construct Corpus object on the basis of it
        # assumes file naming convention: <category>_<title>.txt
        if not os.path.exists(data_home):
            raise IOError("Your corpus folder does not exist...")
        target_names = {}
        texts, titles, targets = [], [], []
        for filename in os.listdir(data_home):
            if filename.startswith("."):
                continue
            if "_" not in filename or not filename.endswith(".txt") or filename.count("_") > 1:
                raise ValueError("Filename: "+filename+" wrongly formatted (should be: category_title.txt)")
            target_name, title = filename.replace(".txt", "").split("_")
            if target_name not in target_names:
                target_names[target_name] = len(target_names)
            target = target_names[target_name]
            with codecs.open(os.path.join(data_home, filename), mode='r', encoding=encoding) as infile:
                text = infile.read()
            if not text.strip():
                raise ValueError("Filename: "+filename+" does not contain any text...")
            texts.append(text)
            titles.append(title)
            targets.append(target)
        target_names = sorted(target_names, key=target_names.__getitem__)
        return Corpus(texts=texts, titles=titles, targets=targets, target_names=target_names)

    def tokenize(self, tokenization_option=""):
        tokenizer = RegexpTokenizer(r'\w+')
        for i, text in enumerate(self.texts):
            self.texts[i] = [word.lower() for word in tokenizer.tokenize(text)][:100000] # punctuation removed here

    def untokenize(self, sep=" "):
        "make strings of samples/texts to extract char-level features"
        for i, text in enumerate(self.texts):
            self.texts[i] = sep.join(text) # punctuation removed here       

    def sample(self, method="slicing", sample_size=5000, step_size=1000):
        # default: slicing (i.e. stylo's normal_sampling (also implement: no_sampling, random_sampling)
        texts, titles, targets = [], [], []
        for text, title, target in zip(self.texts, self.titles, self.targets):
            start_index, end_index = 0, sample_size
            if sample_size > len(text):
                raise ValueError("Title: %s only has %d tokens (< sample_size = %d)" % (title, len(text)))
            sample_counter = 1
            while end_index <= len(text):
                texts.append(text[start_index: end_index])
                titles.append(title + "_" + str(sample_counter))
                targets.append(target)
                sample_counter += 1
                start_index += step_size
                end_index += step_size
        return Corpus(texts=texts, titles=titles, targets=targets, target_names=self.target_names)

    def __len__(self):
        return len(self.texts)

    def __repr__(self):
        return "<Corpus(%s texts)> " % len(self)

    def __str__(self):
        info_string = "This corpus contains:\n"
        for text, title, target in zip(self.texts, self.titles, self.targets):
            info_string+="\t- %s (cat: %s): %r[...]\n" % (title, self.target_names[target], text[:10])
        return info_string
            
