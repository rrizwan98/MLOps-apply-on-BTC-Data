from keras.models import Sequential
# from keras import layers
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import RMSprop

from src.exception import CustomException
from src.logger import logging

def BiGRU_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    return model