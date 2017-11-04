import numpy as np
import skimage.io
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import os
import pickle

import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras import regularizers


def readData(csv):
    x_train = []
    y_train = []
    for path, cls in np.loadtxt(csv, delimiter=',', dtype=str):
        x_train.append(skimage.io.imread(path, as_grey=False))
        y_train.append(int(cls) - 1)
    return np.array(x_train), np.array(y_train)

train_samples_cnt = 20000


def save_bottleneck_features(generator):
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    data = next(generator)
    x = data[0]
    y = data[1]

    x = model.predict(x)
    np.save(open('bottleneck_features_train.npy', 'wb'), x)
    np.save(open('bottleneck_features_train_y.npy', 'wb'), y)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.load(open('bottleneck_features_train_y.npy', 'rb'))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(111, activation='softmax'))

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=150,
              batch_size=256)
    model.save_weights('top_model_weights.h5')


if __name__ == '__main__':
    print('Data reading')
    x_train, y_train = readData('../train.csv')
    y_train = keras.utils.to_categorical(y_train, 111)

    datagen = ImageDataGenerator(rescale=1. / 255,
                                 rotation_range=90,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='reflect'
                                 )

    # i = 0
    # for batch in datagen.flow(x_train, batch_size=1,
    #                           save_to_dir='_preview',
    #                           save_format='jpeg'):
    #     i += 1
    #     if i > 100:
    #         break  # otherwise the generator would loop indefinitely

    # save_bottleneck_features(datagen.flow(x_train, y_train,
    #                                      batch_size=train_samples_cnt))
    train_top_model()

    