import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer


class SingularPluralDetector:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return None

    def is_plural(self, word):
        lemma = self.lemmatizer.lemmatize(word, pos=wn.NOUN)
        if lemma != word:
            return True
        return False

    def detect_singular_plural(self, sentence):
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)

        singulars = []
        plurals = []

        for word, tag in tagged:
            if tag in ['NN', 'NNP']:
                singulars.append(word)
            elif tag in ['NNS', 'NNPS'] or (tag.startswith('N') and self.is_plural(word)):
                plurals.append(word)

        return singulars, plurals
