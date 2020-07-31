import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def Neural_Network():
    model = Sequential()
    neurons_num = 80

    #Input layer
    model.add(Dense(1, input_dim=1, activation='tanh'))
    
    #Hidden layer
    model.add(Dense(neurons_num, activation='tanh'))
    model.add(Dense(neurons_num, activation='tanh'))
    model.add(Dense(neurons_num, activation='tanh'))
    model.add(Dense(neurons_num, activation='tanh'))

    #Output layer
    model.add(Dense(1, activation='sigmoid'))

    return model
