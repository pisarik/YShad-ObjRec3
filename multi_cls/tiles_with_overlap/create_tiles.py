import numpy as np
from skimage import io

import os


def piramidTiling(img, tile_rows=None, tile_cols=None):
    '''
    Tile image into NxM patches + patches on the crosses
    '''
    N = img.shape[0] // tile_rows
    M = img.shape[1] // tile_cols
    img = img[:tile_rows * N, :tile_cols * M]
    tiles = []

    while img.shape[0] != tile_rows and img.shape[1] != tile_cols:
        for i in range(0, img.shape[0], tile_rows):
            for j in range(0, img.shape[1], tile_cols):
                tiles.append(img[i:i + tile_rows, j:j + tile_cols])

        img = img[tile_rows // 2: -tile_rows // 2,
                  tile_cols // 2: -tile_cols // 2]

    return tiles


if __name__ == '__main__':
    samples = np.loadtxt('common_train.csv', delimiter=',', dtype=str)

    result_csv = []
    for path, cls in samples:
        img = io.imread(path, as_grey=True)
        for idx, tile in enumerate(piramidTiling(img, 64, 64)):
            name = os.path.basename(path)
            save_path = os.path.join('_tiles', '{}_{}.png'.format(name, idx))
            io.imsave(save_path, tile)
            result_csv.append([save_path, cls])

    result_csv = np.array(result_csv)
    np.savetxt('tiles.csv', result_csv, header='path,cls', fmt='%s',
               delimiter=',')
