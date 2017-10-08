import numpy as np
import skimage.io
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import os
import pickle

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD


def generateReadme(model):
    readme = open('readme.md', 'w')
    print('# CNN model', file=readme)
    print('### Preprocessing', file=readme)
    print('* One sample splitted into 9 main tiles 64x64. Plus 4 overlapping', file=readme)
    print('  tiles on the edges of main tiles. Then each tile resized to 16x16.', file=readme)
    print('* Dataset of all tiles standardized (centered + scaled).', file=readme)
    print('### Augmentation', file=readme)
    print('90 degrees, [0.5, 2] zoom, reflect', file=readme)
    print('### Architecture', file=readme)
    print('![Architecture](architecture.png)', file=readme)
    print('### Model summary', file=readme)
    model.summary(print_fn=lambda x: readme.write(x + '\n'))
    print('### Results', file=readme)
    print('![Loss plot](loss.png)', file=readme)


def readData(csv):
    x_train = []
    y_train = []
    for path, cls in np.loadtxt(csv, delimiter=',', dtype=str):
        x_train.append(skimage.io.imread(path, as_grey=True))
        y_train.append(int(cls) - 1)
    return np.array(x_train), np.array(y_train)


def preprocessData(train):
    shape = train.shape
    train = train.reshape(shape[0], -1)
    scaler = StandardScaler().fit(train)
    with open('scaler.pkl', 'wb') as handle:
        pickle.dump(scaler, handle)
    return scaler.transform(train).reshape(shape)


if __name__ == '__main__':
    print('Data reading')
    x_train, y_train = readData('train.csv')
    print('Preprocessing')
    x_train = preprocessData(x_train)
    print(x_train.shape)
    y_train = keras.utils.to_categorical(y_train, 111)

    print('Data shapes')
    x_train = x_train.reshape((x_train.shape[0],
                               x_train.shape[1], x_train.shape[2], 1))
    print(x_train.shape, y_train.shape)

    datagen = ImageDataGenerator(rotation_range=90, zoom_range=[0.5, 2],
                                 fill_mode='reflect')

    # i = 0
    # for batch in datagen.flow(x_train, batch_size=1,
    #                           save_to_dir='preview',
    #                           save_format='jpeg'):
    #     i += 1
    #     if i > 100:
    #         break  # otherwise the generator would loop indefinitely

    model = Sequential()
    # input: 64x64 images with 1 channel -> (64, 64) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(16, (3, 3), activation='relu',
                     input_shape=x_train.shape[1:]))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(111, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    print(model.summary())

    history = model.fit(x_train, y_train, batch_size=32, epochs=25)

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')

    plt.savefig('loss.png')
    model.save('model.h5')
    os.environ["PATH"] += (os.pathsep +
                           'C:/Program Files (x86)/Graphviz2.38/bin/')
    keras.utils.plot_model(model, to_file='architecture.png', show_shapes=True)

    generateReadme(model)
