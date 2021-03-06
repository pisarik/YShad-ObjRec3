import numpy as np
import pickle
from collections import Counter
from scipy.stats.mstats import gmean

import skimage.io

import tensorflow as tf
import keras

from os import path

from create_tiles import piramidTiling


if __name__ == '__main__':
    model_name = 'pretrained_vgg'
    # with open(path.join(model_name, 'scaler.pkl'), 'rb') as handle:
    #     scaler = pickle.load(handle)
    model = keras.models.load_model(path.join(model_name,
                                              'model.h5'))

    path_to_data = '../../_data'
    path_to_csv = path.join(path_to_data, 'brodatz_dataset_test_submit.csv')
    test_samples = np.loadtxt(path_to_csv, skiprows=1,
                              delimiter=',', dtype=str)
    submission = [['path', 'class']]
    for ind, (img_path, cls) in enumerate(test_samples):
        print(ind)
        full_path = path.join(path_to_data, img_path)
        img = skimage.io.imread(full_path, as_grey=False)
        tiles = piramidTiling(img, 64, 64)

        for idx, tile in enumerate(tiles):
            shape = tile.shape
            tile = tile.astype(np.float)
            tile /= 255
            tile = np.dstack((tile, tile, tile))
            # tile = tile.ravel()
            # tile = scaler.transform(tile)
            tiles[idx] = tile  # .reshape(shape)

        tiles = np.array(tiles)
        # tiles = tiles.reshape((tiles.shape[0],
        #                        tiles.shape[1], tiles.shape[2], 1))

        preds = model.predict(tiles)
        pred = np.argmax(gmean(preds)) + 1

        submission.append([img_path, str(pred)])

    path_to_save = path.join(model_name, 'submission.txt')
    np.savetxt(path_to_save, submission, fmt='%s', delimiter=',')
