import numpy as np
from skimage import io
import cv2

import os


def piramidTiling(img, tile_rows=None, tile_cols=None):
    '''
    Tile image into NxM patches + patches on the crosses
    '''
    N = img.shape[0] // tile_rows
    M = img.shape[1] // tile_cols
    img = img[:tile_rows * N, :tile_cols * M]
    tiles = []

    # while img.shape[0] != tile_rows and img.shape[1] != tile_cols:
    for k in range(2):
        for i in range(0, img.shape[0], tile_rows):
            for j in range(0, img.shape[1], tile_cols):
                tile = img[i:i + tile_rows, j:j + tile_cols]
                # tile = cv2.resize(tile, (32, 32), interpolation=cv2.INTER_AREA)
                tiles.append(tile)

        img = img[tile_rows // 2: -tile_rows // 2,
                  tile_cols // 2: -tile_cols // 2]

    return tiles


if __name__ == '__main__':
    samples = np.loadtxt('common_train.csv', delimiter=',', dtype=str)

    os.makedirs('_tiles', exist_ok=True)

    result_csv = []
    for path, cls in samples:
        img = io.imread(path, as_grey=True)
        for idx, tile in enumerate(piramidTiling(img, 32, 32)):
            name = os.path.basename(path)
            save_path = os.path.join('_tiles', '{}_{}.png'.format(name, idx))
            io.imsave(save_path, tile)
            result_csv.append([os.path.join('..', save_path), cls])

    result_csv = np.array(result_csv)
    np.savetxt('train.csv', result_csv, header='path,cls', fmt='%s',
               delimiter=',')
