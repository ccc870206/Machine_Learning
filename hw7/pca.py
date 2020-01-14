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


def PCA(datas, k, kernel=None):
    if callable(kernel):
        datas = kernel(datas, datas)
        N = datas.shape[0]
        N1 = np.ones((N, N)) / N
        S = (datas
             - np.matmul(N1, datas)
             - np.matmul(datas, N1)
             + np.matmul(np.matmul(N1, datas), N1)
             ) / N
    else:
        S = covariance(datas.T, datas.T)

    value, vector = np.linalg.eigh(S)


    # value = load_pkl_data('/Users/yen/Downloads/hw7/eval_25')
    # vector = load_pkl_data('/Users/yen/Downloads/hw7/evec_25')

    # showEigen(value, vector, isMin=False)

    start_idx = 0
    max_idxs = np.flip(np.argsort(value))
    W = np.concatenate([
        vector[:, max_idxs[i]][:, None]
        for i in range(start_idx, k + start_idx)
    ], axis=1)

    # whitening
    if callable(kernel):
        W /= np.sqrt(value[max_idxs[start_idx:start_idx + k]])[None, :]

    return np.matmul(datas, W), W, value, vector


def showFaces(faces, col=10):
    f = faces.reshape(-1, 100, 100)
    n = f.shape[0]
    all_faces = []
    for i in range(int(n / col)):
        all_faces += [np.concatenate(f[col * i:col * (i + 1)], axis=1)]

    all_faces = np.concatenate(all_faces, axis=0)
    plt.figure(figsize=(4 * (n / col), 4 * col))
    plt.imshow(all_faces, cmap='gray')
    plt.show()


def pca_face(X, no_dims=15):
    X_mean = np.mean(X, axis=0)
    diff_X = X - X_mean

    # cov_mat = np.matmul(diff_X, diff_X.T)

    cov_mat = np.cov(diff_X, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    print(eigen_vectors)
    topk_eigen_values = np.argsort(eigen_values)[: -(no_dims + 1): -1]
    matrix_w = eigen_vectors[:, topk_eigen_values]
    print(matrix_w)
    # eigenspace and eigenvector of D.TD
    # eigen_values_D, eigen_vectors_D = np.linalg.eig(cov_mat)

    # eigenspace and eigenvector of covariance
    # eigen_vectors = np.matmul(diff_X.T, eigen_vectors_D)

    fig = plt.figure(figsize=(20, 20))
    for i in range(15):
        img = matrix_w[:, i].reshape(100, 100)
        fig.add_subplot(5, 5, i + 1)
        plt.axis('off')
        plt.imshow(img, cmap='gray')
    plt.savefig('eigenface.jpg')

    X_test = X[np.random.choice(X.shape[0], 10, replace=False)]

    test_feature = np.matmul(X_test, matrix_w)
    test_restore = np.matmul(test_feature.T, X_test) + X_mean

    fig = plt.figure(figsize=(20, 20))
    for i in range(10):
        img = test_restore[i].reshape(100, 100)
        fig.add_subplot(2, 5, i + 1)
        plt.axis('off')
        plt.imshow(img, cmap='gray')
    plt.savefig('face_restore.jpg')


def linear_kernel(x):
    return np.matmul(x, x.T)


# with PIL.Image.open(
#         './Yale_Face_Database/Training/subject01.centerlight.pgm'
# ) as im:
#     print(np.array(im).shape)

#
start = timeit.default_timer()
if __name__ == '__main__':
    # load_data(dir_test, 'test')


    image_train = load_pkl_data('train').astype('int')
    image_test = load_pkl_data('test')


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

    k = linear_kernel(image25_center)
    N1 = np.ones((image25_center.shape[0],image25_center.shape[0] ))/25
    c = k - np.matmul(k, N1) - np.matmul(N1, k) + np.matmul(N1, np.matmul(k, N1))


    print(c)
    # print(c, c.shape)
    save_pkl_data(c, 'cov_kernel')

    #print(c)
    
    e_value, e_vector = np.linalg.eigh(c)

    save_pkl_data(e_value, 'eval_kernel')
    save_pkl_data(e_vector, 'evec_kernel')
    #e_value = load_pkl_data('eval_25_new')
    #e_vector = load_pkl_data('evec_25_new')

    #print(e_value)
    #print(e_vector)
    
    sorted_index = np.argsort(e_value)[::-1]
    e_vector = e_vector[:, sorted_index]

    save_pkl_data(e_vector, 'evec_kernel_sort')
    #e_vector = load_pkl_data('evec_25_new')
    #save_pkl_data(e_vector, 'evec_25_new_sort')

    
    
    #e_vector = load_pkl_data('evec_25_new_sort')
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
        #cv2.imwrite('./figure_{}.png'.format(idx), pcd*25)

        #print(pcd*255)
    #print(e_vector)

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
        #cv2.imwrite('./new_img_{}.png'.format(i), reconstruct)
    #img_new_space = np.matmul(e_vector.T, image25)
    #print(np.var(img_new_space, axis=1)[:15])
    

    # print(image25_center, image25_center.shape)


    # print(image25.shape)



    # for i in range(25):
    #
    #     plt.imshow(image_train[i], cmap='gray')
    #     print(image_train)
    #     plt.show()


    # print(image_test.shape)
    # image25 = image_train[:25].reshape(25, -1)
    # image25_mean = np.mean(image25, axis=1)
    # c = covariance_matrix(image25)




    # save_pkl_data(c, '/Users/yen/Downloads/hw7/cov_25')
    #e_value, e_vector = np.linalg.eig(c)

    #save_pkl_data(e_value, 'eval_25')
    #save_pkl_data(e_vector, 'evec_25')
    # e_value = load_pkl_data('/Users/yen/Downloads/hw7/eval_25')
    # e_vector = load_pkl_data('/Users/yen/Downloads/hw7/evec_25')

    #
    # face_pcaspace, face_eigen, face_eva, face_evc = PCA(image25, 25)
    # print(face_eigen)
    # print(face_eigen.shape)
    # showFaces(face_eigen.T, col=5)



    #
    # n_evector = 2
    # sorted_index = np.argsort(e_value)[::-1]
    # e_vector = e_vector[:, sorted_index].astype(np.float)
    #
    # # index = np.argsort(e_value)[-2:][::-1]
    # pc = e_vector[:, :n_evector]
    #
    # pc2 = e_vector[:, 1]
    # # print(pc2 + image25_mean)
    #
    #
    # for idx in range(15):
    #     pcd = e_vector[:, idx]# + image25_mean
    #     pcd = np.reshape(pcd, (100, 100)) / np.max(pcd)
    #
    #     cv2.imwrite('/Users/yen/Downloads/hw7/figure_{}.png'.format(idx), pcd * 255)
    #
    #
    # img_new_space = np.matmul(e_vector.T, image25)
    # print(np.var(img_new_space, axis=1)[:15])


    stop = timeit.default_timer()
    print('Time: ', stop - start)
