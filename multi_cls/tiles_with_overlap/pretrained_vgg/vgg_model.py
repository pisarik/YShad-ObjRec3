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


def formatSummaryLine(line):
    if '___' in line:
        return ''
    elif len(line) < 30:
        return line + '  \n'
    else:
        sep_poses = [28, 52]
        for i, pos in enumerate(sep_poses):
            line = line[:pos + i] + '|' + line[pos + i:]

        return line.replace('=', '-') + '\n'


def generateReadme(model):
    readme = open('readme.md', 'w')
    print('# CNN model', file=readme)
    print('### Preprocessing', file=readme)
    print('* One sample splitted into 9 main tiles 64x64. Plus 4 overlapping', file=readme)
    print('  tiles on the edges of main tiles. Then each tile resized to 32x32.', file=readme)
    print('* Dataset of all tiles standardized (centered + scaled).', file=readme)
    print('### Augmentation', file=readme)
    print('90 degrees, [0.5, 2] zoom, reflect', file=readme)
    print('### Architecture', file=readme)
    print('![Architecture](architecture.png)', file=readme)
    print('### Model summary', file=readme)
    model.summary(print_fn=lambda x: readme.write(formatSummaryLine(x)))
    print('### Results', file=readme)
    print('![Loss plot](loss.png)', file=readme)


def readData(csv):
    x_train = []
    y_train = []
    for path, cls in np.loadtxt(csv, delimiter=',', dtype=str):
        x_train.append(skimage.io.imread(path, as_grey=False))
        y_train.append(int(cls) - 1)
    return np.array(x_train), np.array(y_train)


def save_bottlebeck_features(datagen, train_samples_cnt):
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    bottleneck_features_train = model.predict_generator(
        datagen, train_samples_cnt // 16)
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)


if __name__ == '__main__':
    print('Data reading')
    x_train, y_train = readData('../train.csv')
    y_train = keras.utils.to_categorical(y_train, 111)

    print('Data shapes')
    # x_train = x_train.reshape((x_train.shape[0],
    #                            x_train.shape[1], x_train.shape[2], 1))
    print(x_train.shape, y_train.shape)

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

    vgg16 = applications.VGG16(include_top=False, weights='imagenet',
                               input_tensor=Input(x_train.shape[1:]))
    top_model = Sequential()
    # don't change vgg16
    # for layer in vgg16.layers:
    #     layer.trainable = False

    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(111, activation='sigmoid'))
    # top_model.add(Dense(256, activation='relu',
    #                 kernel_regularizer=regularizers.l2(0.001)))
    # top_model.add(Dropout(0.5))
    # top_model.add(Dense(111, activation='softmax',
    #                 kernel_regularizer=regularizers.l1(0.001)))
    top_model.load_weights('top_model_weights.h5')

    model = Sequential()
    for layer in vgg16.layers:
        model.add(layer)
    for layer in top_model.layers:
        model.add(layer)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam())
    # print(model.summary())
    # keras.utils.plot_model(model, to_file='architecture.png', show_shapes=True)

    # # model = keras.models.load_model('model.h5')
    # history = model.fit_generator(datagen.flow(x_train, y_train,
    #                                            batch_size=256),
    #                               steps_per_epoch=len(x_train) / 256,
    #                               epochs=150)

    # plt.figure(figsize=(12, 6))
    # plt.plot(history.history['loss'])
    # # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'valid'], loc='upper left')

    # plt.savefig('loss.png')
    model.save('model.h5')
    os.environ["PATH"] += (os.pathsep +
                           'C:/Program Files (x86)/Graphviz2.38/bin/')
    keras.utils.plot_model(model, to_file='architecture.png', show_shapes=True)

    # plt.show()
    # generateReadme(model)
