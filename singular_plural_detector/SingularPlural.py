import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

# Need to download the following in order to make the code work, can be removed if already downloaded
'''
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
'''

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
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

def is_plural(word, pos_tag):
    lemma = lemmatizer.lemmatize(word, pos=wn.NOUN)
    if lemma != word:
        return True
    return False

def detect_singular_plural(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)

    singulars = []
    plurals = []

    for word, tag in tagged:
        if tag in ['NN', 'NNP']:
            singulars.append(word)
        elif tag in ['NNS', 'NNPS'] or (tag.startswith('N') and is_plural(word, tag)):
            plurals.append(word)

    return singulars, plurals

if __name__ == "__main__":
    toFind = "men, woman, goose, geese, stars, eyes, students!"
    singulars, plurals = detect_singular_plural(toFind)
    print("Singulars:", singulars)
    print("Plurals:", plurals)
