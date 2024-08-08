from flask import Flask, request, jsonify
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'\bnot\b', 'not_', text)
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'review' not in data:
        return jsonify({"error": "No review provided"}), 400

    review = data['review']
    processed_review = preprocess_text(review)
    vectorized_review = vectorizer.transform([processed_review])
    prediction = model.predict(vectorized_review)

    sentiments = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    return jsonify({"sentiment": sentiments[prediction[0]]})

if __name__ == "__main__":
    app.run(debug=True)
