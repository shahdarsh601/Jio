import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re

from sklearn.utils import resample

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Loadign the dataset..
df = pd.read_csv('IMDB Dataset.csv', encoding='ISO-8859-1')


# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'\bnot\b', 'not_', text)
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

# Applying preprocessing
df['review'] = df['review'].apply(preprocess_text)

# Label encoding
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1, 'neutral': 2})

# Separating each class for balancing
negative = df[df['sentiment'] == 0]
positive = df[df['sentiment'] == 1]
neutral = df[df['sentiment'] == 2]

# Resampling to balance the classes
neutral_upsampled = resample(neutral,
                             replace=True,  # sample with replacement
                             n_samples=len(negative),  # match number in majority class
                             random_state=42)  # reproducible results

# Combining majority class with upsampled minority class
df_balanced = pd.concat([negative, positive, neutral_upsampled])

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(df_balanced['review'], df_balanced['sentiment'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Model training
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(X_train_vectorized, y_train)

# Saving the model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# accuracy score
y_pred = model.predict(X_test_vectorized)

# Print the classification report and accuracy score
labels = [0, 1, 2]  # Match these with the label encoding used
target_names = ['Negative', 'Positive', 'Neutral']
print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
