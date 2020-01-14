import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os
import pickle as pkl
import timeit
import cv2

from scipy.spatial.distance import pdist, squareform

import PIL.Image


# np.set_printoptions(threshold=np.inf)

dir_train = '/Users/yen/Downloads/Yale_Face_Database_compress/Training'
dir_test = '/Users/yen/Downloads/Yale_Face_Database_compress/Testing'

filename = dir_train+'subject01.centerlight.pgm'


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

        # img = cv2.imread(filename, 0)
        return img


def load_data(dir, type):
    filelist = sorted(os.listdir(dir))
    # print(filelist)
    lenth = len(filelist)
    img_list = np.zeros((lenth, 100, 100))
    # print(lenth)
    for i in range(lenth):
        img = read_image(os.path.join(dir, filelist[i]))
        img_list[i, :] = img

    save_pkl_data(img_list, type)
    return img_list


def load_label(dir, type):
    filelist = sorted(os.listdir(dir))

    # print(filelist)
    label = [int(filename[7:9]) for filename in filelist]

    save_pkl_data(label, 'label_'+type)

    return label



def save_pkl_data(arr, filename):
    with open('./'+filename+'.pkl', 'wb') as f:
        pkl.dump(arr, f)


def load_pkl_data(filename):
    with open(filename+'.pkl', 'rb') as f:
        w = pkl.load(f)
    return w


def rbf_kernel(x, gamma):
    return squareform(np.exp(-gamma * pdist(x, 'sqeuclidean')))


def linear_kernel(x):
    return np.matmul(x, x.T)


def covariance_matrix(matrix, kernel = 0):
    matrix_mean = np.mean(matrix, axis=1).reshape(-1, 1)
    matrix_center = (matrix - matrix_mean)

    if kernel == 0:
        return np.cov(matrix_center)

    else:
        k = linear_kernel(matrix_center)
        N1 = np.ones((matrix_center.shape[0],matrix_center.shape[0]))/25
        c = k - np.matmul(k, N1) - np.matmul(N1, k) + np.matmul(N1, np.matmul(k, N1))
        return c


def eigenface(w, dim):
    fig = plt.figure(figsize=(20, 20))
    for idx in range(dim):
        pcd = w[:, idx]
        pcd = np.reshape(pcd, (100, 100))
        fig.add_subplot(5, 5, idx + 1)
        plt.imshow(pcd, cmap='gray')
    plt.savefig('./eigenface.png')
    plt.clf()


def reconstruction(w, matrix):
    fig = plt.figure(figsize=(20, 20))
    for i in range(10):
        new_space = np.matmul(w, w.T)
        reconstruct = np.matmul(new_space, matrix[:, i]).reshape(100, 100)

        fig.add_subplot(5, 4, i * 2 + 1)
        plt.imshow(matrix[:, i].reshape(100, 100), cmap='gray')
        fig.add_subplot(5, 4, i * 2 + 2)
        plt.imshow(reconstruct, cmap='gray')
    plt.savefig('./restore.png')
    plt.clf()


start = timeit.default_timer()
if __name__ == '__main__':
    image_train = load_data(dir_train, 'train')
    image_test = load_data(dir_test, 'test')
    label_train = load_label(dir_train, 'train')
    label_test = load_label(dir_test, 'test')


    dim = 25

    # first 25 images
    image25 = image_train[:25].reshape(25, -1).T

    c = covariance_matrix(image25)

    # compute eigenvalue, eigenvector
    e_value, e_vector = np.linalg.eigh(c)
    sorted_index = np.argsort(e_value)[::-1]
    e_vector = e_vector[:, sorted_index]

    # choose principle component
    w = e_vector[:, :dim + 1]
    # draw eigenface
    eigenface(w, dim)

    # random choose 10 images
    random_index = np.random.choice(25, 10)
    image10 = image25[:, random_index]
    # reconstruct the face
    reconstruction(w, image10)

    # face recognition
    train = image_train.reshape(135, -1).T
    test = image_test.reshape(30, -1).T

    r_train = np.matmul(w.T, train)
    r_test = np.matmul(w.T, test)

    acc = 0
    for i in range(30):
        target = r_test[:, i].reshape(-1, 1)
        diff = r_train - target
        distance = np.linalg.norm(diff, axis=0)
        idx = np.argmin(distance)
        print("predict:", label_train[idx])
        print("truth:", label_test[i])
        if label_train[idx] == label_test[i]:
            acc +=1
    print("wrong", 30-acc, "/", 30)
    print("acc:", acc/30*100, "%")


    stop = timeit.default_timer()
    print('Time: ', stop - start)
