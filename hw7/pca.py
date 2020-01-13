import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
import timeit
import cv2



from scipy.spatial.distance import pdist, squareform

# np.set_printoptions(threshold=np.inf)

dir_train = '/Users/yen/Downloads/Yale_Face_Database_compress/Training'
dir_test = '/Users/yen/Downloads/Yale_Face_Database_compress/Testing'

filename = dir_train+'subject01.centerlight.pgm'


def read_image(filename):
    with open(filename, 'rb') as f:
        # magic_number = f.readline().strip().decode('utf-8')
        # info = f.readline().strip().decode('utf-8').split(' ')
        # width, height = f.readline().strip().decode('utf-8').split(' ')
        # width = int(width)
        # height = int(height)
        # maxval = f.readline().strip().decode('utf-8')
        #
        # img = np.zeros((height, width))
        # img[:, :] = [[ord(f.read(1)) for j in range(width)] for i in range(height)]
        # # print(img.shape)
        # # img = cv2.resize(img, (100, 100))
        # # img/=255
        # imgplot = plt.imshow(img, cmap='gray')
        # plt.show()

        img = cv2.imread(filename, 0)
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
    # print(img_list)
    # print(img_list.shape)
    save_pkl_data(img_list, type)



def save_pkl_data(arr, filename):
    with open('./'+filename+'.pkl', 'wb') as f:
        pkl.dump(arr, f)


def load_pkl_data(filename):
    with open(filename+'.pkl', 'rb') as f:
        w = pkl.load(f)
    return w


def similarity_matrix(x):
    # d_coord = pdist(x, 'sqeuclidean')
    return squareform(pdist(x, 'sqeuclidean'))


def linear(x):
    return np.matmul(x, x.T)


def covariance_matrix(matrix):
    # c = np.cov(matrix)

    avg = np.mean(matrix, axis=0)
    diff = matrix - avg
    result = np.matmul(diff.T, diff) / 25
    print(result)
    print(result.shape)
    return result

start = timeit.default_timer()
if __name__ == '__main__':
    # load_data(dir_test, 'test')

    image_train = load_pkl_data('./train')

    image_test = load_pkl_data('./test')
    # print(image_test.shape)

    # image25 = image_train[:25].reshape(25, -1)
    # c = covariance_matrix(image25)

    # save_pkl_data(c, 'cov_25')
    # e_value, e_vector = np.linalg.eig(c)
    # save_pkl_data(e_value, 'eval_25')
    # save_pkl_data(e_vector, 'evec_25')

    e_value = load_pkl_data('/Users/yen/Downloads/hw7/eval_25')
    e_vector = load_pkl_data('/Users/yen/Downloads/hw7/evec_25')
    # print(e_value)
    # print(e_vector)
    # print(e_vector.shape)
    #
    n_evector = 2
    index = np.argsort(e_value)[1:n_evector+1][::-1]
    w = e_vector[:, index]


    # # index = np.argsort(e_value)[1:n_evector + 1]
    # index = np.argsort(e_value)[-2:][::-1]
    # pc = e_vector[:, index]
    # # print(e_value[index])
    # # print(pc)
    # # print(c)
    # # print(c.shape)


    # a = np.array([[1,2],[3,4]])
    # print(np.mean(a, axis=0))
    # #
    # print(a - np.mean(a, axis=0))
    stop = timeit.default_timer()
    print('Time: ', stop - start)
