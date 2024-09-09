#!/usr/bin/env python3
import os.path
import sklearn.preprocessing
import sklearn.utils
import sklearn.model_selection
import sklearn.metrics
import pandas as pd
import numpy as np
from tensorflow import keras as keras
import config

np.set_printoptions(suppress=True)

gLabelOp = sklearn.preprocessing.LabelEncoder()

csv_header = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'op', 'res']
# csv_types = {
#     'b1': np.int8, 
#     'b2': np.int8, 
#     'b3': np.int8, 
#     'b4': np.int8, 
#     'b5': np.int8, 
#     'b6': np.int8, 
#     'b7': np.int8, 
#     'b8': np.int8, 
#     'b9': np.int8, 
#     'b10': np.int8, 
#     'op':str, 
#     'res': np.int8
# }

def train():
    data = pd.read_csv('/home/zhangjun/tmp/logic_reg/data.csv', names=csv_header)

    x = data.drop(labels=['res'], axis=1)

    y = data['res']

    x['op'] = gLabelOp.fit_transform(x['op'])

    # scaler = sklearn.preprocessing.StandardScaler()
    # x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.25)

    print(f'x shape:{x_train.shape}')
    
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, activation='relu', input_dim=x_train.shape[1]))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    logdir = os.path.join(config.RootDir, 'log')
    tensorboard_cb = keras.callbacks.TensorBoard(logdir)
    
    results = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test), callbacks=[tensorboard_cb])
    print(f'loss:{results.history["loss"]}')
    print(f'accuracy:{results.history["accuracy"]}')
    print(f'val loss:{results.history["val_loss"]}')
    print(f'val accuracy:{results.history["val_accuracy"]}')
    
    test_results = model.evaluate(x_test, y_test)
    print(f'test loss:{test_results[0]}')
    print(f'test accuracy:{test_results[1]}')
    
    y_pred = model.predict_on_batch(x_test)
    
    y_pred = np.where(y_pred < 0.5, 0, 1)
    
    cf = sklearn.metrics.confusion_matrix(np.expand_dims(y_test, axis=1), y_pred)
    print(f'confusion matrix:{cf}')
    
    accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(f'accuracy score:{accuracy_score}')
    
    return model


def predict(model):
    data = pd.read_csv('/home/zhangjun/tmp/logic_reg/test_data.csv', names=csv_header)

    x = data.drop(labels=['res'], axis=1)

    y = data['res']
    
    x['op'] = gLabelOp.fit_transform(x['op'])

    y_pred = model.predict_on_batch(x)
    
    y_pred = np.where(y_pred < 0.5, 0, 1)
    
    x['op'] = gLabelOp.inverse_transform(x['op'])
    
    results = np.hstack((x, np.expand_dims(y, axis=1), y_pred))
    print(results)
    

if __name__ == '__main__':
    model = train()
    
    print('---------------------------- predict ----------------------------')
    
    predict(model)