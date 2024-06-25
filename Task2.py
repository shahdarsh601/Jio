from flask import Flask, request, jsonify
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer


app = Flask(__name__)
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


def is_plural(word):
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
        elif tag in ['NNS', 'NNPS'] or (tag.startswith('N') and is_plural(word)):
            plurals.append(word)

    return singulars, plurals


@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    if 'sentence' not in data:
        return jsonify({"error": "No sentence provided"}), 400

    sentence = data['sentence']
    singulars, plurals = detect_singular_plural(sentence)

    return jsonify({"singulars": singulars, "plurals": plurals})


if __name__ == "__main__":
    app.run(debug=True)
