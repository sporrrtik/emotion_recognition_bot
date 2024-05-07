from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import pickle
from keras.utils import pad_sequences
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np

max_text_len = 20  # длина входного вектора
num_words = 10000

model_lstm = Sequential()
model_lstm.add(Embedding(num_words, 128, input_length=max_text_len))
model_lstm.add(LSTM(16))
model_lstm.add(Dense(units=16, activation='relu'))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(7, activation='softmax'))
# model_lstm.summary()

""" Если запускать этот файл, поменять все пути на ./..."""
model_lstm.load_weights('models/text_lstm_weight_1.h5')

with open('models/encoder_lstm.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

noise = stopwords.words('russian') + list(punctuation)


def remove_stop_words(text, stopwords):
    new_text = []
    for sent in text:
        s = sent.split()
        table = str.maketrans("", "", punctuation)
        new_s = [w.translate(table) for w in s if not w.lower() in stopwords]
        new_s = " ".join(new_s)
        new_text.append(new_s)
    return new_text


def lem_data(text):
    lemmatizer = WordNetLemmatizer()
    new_text = []
    for sent in text:
        s = sent.split()
        new_s = [lemmatizer.lemmatize(word.lower()) for word in s]
        new_s = " ".join(new_s)
        new_text.append(new_s)
    return new_text


def make_array_from_text(text, tokenizer, max_text_len):
    noise = stopwords.words('russian') + list(punctuation)
    text = remove_stop_words(text, noise)
    text = lem_data(text)
    sequences = tokenizer.texts_to_sequences(text)
    text = pad_sequences(sequences, maxlen=max_text_len)

    return text


with open('models/tokenizer_lstm.pickle', 'rb') as f:
    tokenizer = pickle.load(f)


def predict_from_text(user_text):
    arr = np.array([user_text])
    # arr = np.array(["Со мной все хорошо, завтра заеду","Жаль, что так вышло", "Фуу, это отвратительно!"])
    text = make_array_from_text(arr, tokenizer, max_text_len)
    pred = model_lstm.predict(text)
    return encoder2.inverse_transform(pred)
