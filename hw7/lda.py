import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
import timeit
import cv2


dir_train = '/Users/yen/Downloads/Yale_Face_Database_compress/Training'
dir_test = '/Users/yen/Downloads/Yale_Face_Database_compress/Testing'

filename = dir_train+'subject01.centerlight.pgm'


def save_pkl_data(arr, filename):
    with open('./'+filename+'.pkl', 'wb') as f:
        pkl.dump(arr, f)


def load_pkl_data(filename):
    with open(filename+'.pkl', 'rb') as f:
        w = pkl.load(f)
    return w


def read_image(filename):
    with open(filename, 'rb') as f:
        magic_number = f.readline().strip().decode('utf-8')
        info = f.readline().strip().decode('utf-8').split(' ')
        width, height = f.readline().strip().decode('utf-8').split(' ')
        width = int(width)
        height = int(height)
        maxval = f.readline().strip().decode('utf-8')

        img = np.zeros((height, width))
        img[:, :] = [[ord(f.read(1)) for j in range(width)] for i in range(height)]
        # print(img.shape)
        img = cv2.resize(img, (100, 100))
        img/=255
        # imgplot = plt.imshow(img, cmap='gray')
        # plt.show()

        return img


def load_data(dir, type):
    filelist = sorted(os.listdir(dir))

    lenth = len(filelist)
    img_list = np.zeros((lenth, 100, 100))
    # print(lenth)
    for i in range(lenth):
        img = read_image(os.path.join(dir, filelist[i]))
        img_list[i, :] = img

    save_pkl_data(img_list, type)

    return img_list


def load_label(dir, type):
    filelist = sorted(os.listdir(dir))[1:]

    # print(filelist)
    label = [int(filename[7:9]) for filename in filelist]

    save_pkl_data(label, 'label_'+type)

    return label


start = timeit.default_timer()
if __name__ == '__main__':
    image_train = load_pkl_data('train').astype(int)
    image_test = load_pkl_data('test')
    label_train = load_pkl_data('label_train')


    n_label = 3

    image25 = image_train[:25].reshape(25, -1).T
    label25 = np.array(label_train[:25])


    # between-class scatter
    m_total = np.mean(image25, axis=1).reshape(-1,1)

    m_cluster = np.zeros((10000, n_label))
    n_c = np.zeros(n_label)
    for j in range(n_label):
        c_list = np.where(label25 == j+1)[0]
        c_in_x = image25[:, c_list]

        sum_c = np.sum(c_in_x, axis=1)
        n_c[j] = len(c_list)
        m_cluster[:, j] = sum_c/n_c[j]

    Sb = np.zeros((10000,10000))
    for j in range(n_label):
        diff = m_cluster-m_total
        Sb += n_c[j] * np.matmul(diff, diff.T)


    # within-class scatter
    Sw = np.zeros((10000,10000))
    for j in range(n_label):
        c_list = np.where(label25 == j + 1)[0]
        c_in_x = image25[:, c_list]
        diff = c_in_x - m_cluster[:, j].reshape(-1, 1)
        Sw += np.matmul(diff, diff.T)

    target_matrix = np.matmul(np.linalg.inv(Sw+0.1), Sb)


    e_val, e_vector = np.linalg.eig(target_matrix)

    sorted_index = np.argsort(e_val)[::-1]
    e_vector = e_vector[:, sorted_index]

    stop = timeit.default_timer()
    print('Time: ', stop - start)

