import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re
from sentence_transformers import SentenceTransformer
from sklearn.utils import resample

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

print("NLt K data downloaded/")

# Load the dataset
df = pd.read_csv('IMDB Dataset.csv', encoding='ISO-8859-1')
print("Dataset has been loaded")

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'\bnot\b', 'not_', text)
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

# Apply preprocessing with progress tracking
def preprocess_and_track(df):
    total = len(df)
    print(f"Total reviews to process: {total}")
    for i, review in enumerate(df['review']):
        df.at[i, 'review'] = preprocess_text(review)
        if i % 1000 == 0:
            print(f"Processed {i+1}/{total} reviews")

preprocess_and_track(df)
print("Preprocessing completed")

# Label encoding
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1, 'neutral': 2})

# Separate each class for balancing
negative = df[df['sentiment'] == 0]
positive = df[df['sentiment'] == 1]
neutral = df[df['sentiment'] == 2]

# Resample to balance the classes
neutral_upsampled = resample(neutral,
                             replace=True,  # sample with replacement
                             n_samples=len(negative),  # match number in majority class
                             random_state=42)  # reproducible results

# Combine majority class with upsampled minority class
df_balanced = pd.concat([negative, positive, neutral_upsampled])
print("Classes balanced")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df_balanced['review'], df_balanced['sentiment'], test_size=0.2, random_state=42)
print("Data split into train and test sets")

# Load the pre-trained sentence transformer model
model_name = 'distilbert-base-nli-mean-tokens'
embedder = SentenceTransformer(model_name)
print(f"Sentence transformer model '{model_name}' loaded")

# Function to encode sentences in batches
def encode_sentences_in_batches(sentences, batch_size=100):
    embeddings = []
    total = len(sentences)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_embeddings = embedder.encode(sentences[start:end])
        embeddings.extend(batch_embeddings)
        print(f"Encoded {end}/{total} sentences")
    return embeddings

# Create sentence embeddings in batches
X_train_embeddings = encode_sentences_in_batches(X_train.tolist())
X_test_embeddings = encode_sentences_in_batches(X_test.tolist())
print("Sentence embeddings created")

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_embeddings, y_train)
print("Model trained")

# Make predictions and evaluate the model
y_pred = model.predict(X_test_embeddings)
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save the model and embedder
joblib.dump(model, 'sentiment_model.pkl')
embedder.save('sentence_transformer_model/')
print("Model and embedder saved")
