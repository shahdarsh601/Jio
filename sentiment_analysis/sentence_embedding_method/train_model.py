import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


df = pd.read_csv('IMDB Dataset.csv', encoding='ISO-8859-1')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'\bnot\b', 'not_', text)
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)
df['review'] = df['review'].apply(preprocess_text)

#encoding of labels
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1, 'neutral': 2})

# padding and some otkenization
max_vocab_size = 20000
max_sequence_length = 100

tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(df['review'])

X = tokenizer.texts_to_sequences(df['review'])
X = pad_sequences(X, maxlen=max_sequence_length)

y = df['sentiment'].values

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# defining the neural network with an embedding layer
embedding_dim = 50

model = Sequential([
    Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training this model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

#evaluating this model
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Loss: {loss}, Accuracy: {accuracy}')

# Save the model
model.save('sentiment_analysis_model.h5')
