import numpy as np
import pickle
import librosa
import soundfile as sf

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Flatten, BatchNormalization

model_cnn = Sequential()
model_cnn.add(Input(shape=(20, 1)))
model_cnn.add(Conv1D(32, 5, padding='valid', activation='relu'))
model_cnn.add(BatchNormalization())
model_cnn.add(Conv1D(64, 5, padding='valid', activation='relu'))
model_cnn.add(BatchNormalization())
model_cnn.add(Conv1D(128, 5, padding='valid', activation='relu'))
model_cnn.add(GlobalMaxPooling1D())
model_cnn.add(Dropout(0.3))
model_cnn.add(Flatten())
model_cnn.add(Dense(units=128, activation='relu', kernel_regularizer='l2'))
model_cnn.add(Dense(units=64, activation='relu', kernel_regularizer='l2'))
model_cnn.add(Dense(7, activation='softmax'))
# model_cnn.summary()

""" Если запускать этот файл, поменять все пути на ./..."""
model_cnn.load_weights("models/audio_weight_1.h5", skip_mismatch=False)


def get_mfcc_from_audio(audio_file_path):
    d, samplerate = sf.read(audio_file_path)
    d = np.mean(librosa.feature.mfcc(y=d).T, axis=0)

    return d


# audio_file_path = "./myfile.wav"
with open('models/encoder_cnn.pickle', 'rb') as f:
    encoder2 = pickle.load(f)


def predict_from_audio(audio_file_path):
    mfcc = get_mfcc_from_audio(audio_file_path)
    pred = model_cnn.predict(np.array([mfcc]))
    return encoder2.inverse_transform(pred)
