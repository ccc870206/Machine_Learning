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


def similarity_matrix(x):
    # d_coord = pdist(x, 'sqeuclidean')
    return squareform(pdist(x, 'sqeuclidean'))


def linear(x):
    return np.matmul(x, x.T)


def covariance_matrix(matrix):
    # c = np.cov(matrix)

    avg = np.mean(matrix, axis=1)
    diff = matrix - np.expand_dims(avg, axis=-1)
    result = np.matmul(diff, diff.T) / 25
    # print(result)
    # print(result.shape)
    return result


def covariance(x, y):
    x1 = x - (np.sum(x, axis = 1) / x.shape[1])[:,None]
    y1 = y - (np.sum(y, axis = 1) / y.shape[1])[:,None]
    return np.matmul(x1, y1.T)/ x1.shape[1]


def linear_kernel(x):
    return np.matmul(x, x.T)


# with PIL.Image.open(
#         './Yale_Face_Database/Training/subject01.centerlight.pgm'
# ) as im:
#     print(np.array(im).shape)

#
start = timeit.default_timer()
if __name__ == '__main__':
    # image_train = load_data(dir_train, 'train')
    # image_test = load_data(dir_test, 'test')

    # label_train = load_label(dir_train, 'train')
    # label_test = load_label(dir_test, 'test')



    image_train = load_pkl_data('train').astype('int')
    image_test = load_pkl_data('test')

    label_train = load_pkl_data('label_train')
    label_test = load_pkl_data('label_test')
    print(len(label_test))

    dim = 25

    image25 = image_train[:25].reshape(25, -1).T
    #image25 = image_train[:25].reshape(25, -1)
    #pca_face(image25, 25)

    image25_mean = np.mean(image25, axis=1).reshape(-1,1)
    #
    # print(image25, image25.shape)
    # print(image25_mean)
    image25_center = (image25 - image25_mean)

    ###c = np.cov(image25_center)

    ###k = linear_kernel(image25_center)
    ###N1 = np.ones((image25_center.shape[0],image25_center.shape[0] ))/25
    ###c = k - np.matmul(k, N1) - np.matmul(N1, k) + np.matmul(N1, np.matmul(k, N1))


    #print(c)
    # print(c, c.shape)
    ###save_pkl_data(c, 'cov_kernel')

    #print(c)

    ###e_value, e_vector = np.linalg.eigh(c)

    ###save_pkl_data(e_value, 'eval_kernel')
    ###save_pkl_data(e_vector, 'evec_kernel')
    #e_value = load_pkl_data('eval_25_new')
    #e_vector = load_pkl_data('evec_25_new')

    #print(e_value)
    #print(e_vector)

    ###sorted_index = np.argsort(e_value)[::-1]
    ###e_vector = e_vector[:, sorted_index]

    ###save_pkl_data(e_vector, 'evec_kernel_sort')
    #e_vector = load_pkl_data('evec_25_new')
    #save_pkl_data(e_vector, 'evec_25_new_sort')



    e_vector = load_pkl_data('evec_25_new_sort')
    w = e_vector[:, :dim + 1]
    """
    w = e_vector[:,:dim+1]
    print("e_vec", e_vector)
    fig = plt.figure(figsize=(20, 20))
    for idx in range(dim):
        pcd = w[:, idx]
        pcd = np.reshape(pcd, (100, 100))
        fig.add_subplot(5, 5, idx + 1)
        plt.imshow(pcd, cmap='gray')
    plt.savefig('./my_eigenface.png')
    plt.clf()
    """
        #cv2.imwrite('./figure_{}.png'.format(idx), pcd*25)

        #print(pcd*255)
    #print(e_vector)




    """
    random_index = np.random.choice(25, 10)
    image10 = image25[:,random_index]
    print(image10.shape)
    fig = plt.figure(figsize=(20, 20))
    for i in range(10):
        new_space = np.matmul(w, w.T)
        reconstruct = np.matmul(new_space, image10[:,i]).reshape(100, 100)
        
        fig.add_subplot(5, 4, i*2 + 1)
        plt.imshow(image10[:,i].reshape(100, 100), cmap='gray')
        fig.add_subplot(5, 4, i*2 + 2)
        plt.imshow(reconstruct, cmap='gray')
    plt.savefig('./my_restore.png')
    plt.clf()
    """

    train = image_train.reshape(135, -1).T
    test = image_test.reshape(30, -1).T


    new_space = np.matmul(w, w.T)
    r_train = np.matmul(new_space, train)
    r_test = np.matmul(new_space, test)

    # for i in range(30):
    acc = 0
    for i in range(1):
        target = r_test[:, i]
        diff = r_train - target
        distance = np.linalg.norm(diff, axis=0)
        idx = np.argmin(distance)
        print("predict:", label_train[idx])
        print("truth:", label_test[i])
        if label_train[idx] == label_test[i]:
            acc +=1
    print("acc:", acc/30*100, "%")

        # diff = center - tar_matrix[i, :].reshape(1, -1)
        # distance = np.linalg.norm(diff, axis=1)
        # label[i] = np.argmin(distance)



    stop = timeit.default_timer()
    print('Time: ', stop - start)
