#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

def train():
    number_of_points = 100

    x = np.random.uniform(-5, 5, size=(number_of_points, 1))
    print(f'x=\n{x[:5,:]}')

    y = np.random.uniform(-5, 5, size=(number_of_points, 1))
    print(f'y=\n{y[:5,:]}')

    noise = np.random.uniform(-1, 1, size=(number_of_points, 1))
    print(f'noise=\n{noise[:5,:]}')

    z = 5 * x + 3 * y + 2 + noise

    input = np.hstack((x, y))
    print(f'input=\n{input[:5,:]}')

    model = keras.Sequential([
        keras.layers.Dense(units=1, input_shape=[None, 2])])

    model.compile(
        optimizer='sgd', 
        loss='mse', 
        metrics=['mse'])

    history = keras.callbacks.History()

    model.fit(input, z, epochs=100, verbose=1, validation_split=0.2, callbacks=[history])

    print(history.history.keys())

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig('/home/zhangjun/tmp/loss.png')
    plt.close()

    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('metrics')
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig('/home/zhangjun/tmp/mse.png')
    plt.close()

    plt.plot(np.squeeze(model.predict_on_batch(input)), np.squeeze(z))
    plt.title('results')
    plt.xlabel('predicted')
    plt.ylabel('gt')
    plt.savefig('/home/zhangjun/tmp/results.png')
    plt.close()
    
    return model


def predict(model):
    number_of_points = 10
    
    x = np.random.uniform(-10, 10, size=(number_of_points, 1))
    
    y = np.random.uniform(-10, 10, size=(number_of_points, 1))
    
    z = 5 * x + 3 * y + 2
    
    input = np.hstack((x, y))
    
    predicted = model.predict(input)
    
    plt.plot(np.squeeze(predicted), np.squeeze(z))
    plt.title('prediction')
    plt.xlabel('predicted')
    plt.ylabel('gt')
    plt.savefig('/home/zhangjun/tmp/prediction.png')
    plt.close()
    
    data = np.hstack((x, y, z, predicted, predicted - z))
    print(data)
    

if __name__ == '__main__':
    model = train()
    
    predict(model)