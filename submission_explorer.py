import numpy as np
import matplotlib.pyplot as plt
import skimage.io

from os import path


if __name__ == '__main__':
    path_data = '_data'
    e_type = 'multi_cls'
    p_type = 'tiles_with_overlap'
    m_type = 'cnn'

    dataset = np.loadtxt(path.join(path_data, 'brodatz_dataset_train.csv'),
                         skiprows=1, delimiter=',', dtype=str)
    subm = np.loadtxt(path.join(e_type, p_type, m_type, 'submission.txt'),
                      skiprows=1, delimiter=',', dtype=str)

    lst = [0, 5, 50, 200, 500, 800, 1200, 1500, 2000]
    for path_img, cls in subm[lst]:
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('Real')
        path_to_real = dataset[dataset[:, 1] == cls][0, 0]
        real_img = skimage.io.imread(path.join(path_data, path_to_real),
                                     as_grey=True)
        plt.imshow(real_img, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('Classified')
        pred_img = skimage.io.imread(path.join(path_data, path_img),
                                     as_grey=True)
        plt.imshow(pred_img, cmap='gray')
        plt.show()


