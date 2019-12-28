import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import timeit
import pickle as pkl


def similarity_matrix(x, gamma_s, gamma_c):
    coord = np.array([[i, j] for i in range(x.shape[0]) for j in range(x.shape[1])])
    d_coord = -gamma_s * pdist(coord, 'sqeuclidean')
    color = x.reshape(-1, 3)
    d_color = -gamma_c * pdist(color, 'sqeuclidean')
    d_total = d_coord + d_color
    return squareform(np.exp(d_total))


def normalize_L(w):
    D = np.diag(np.sum(w, axis=1))
    D12 = np.diag(1/np.power(np.sum(w, axis=1), 1/2))
    L = np.matmul(np.matmul(D12, (D - w)), D12)
    return L


def unnormalize_L(w):
    D = np.diag(np.sum(w, axis=1))
    L = D - w

    return L


def save_pkl_data(arr, filename):
    with open('/Users/yen/Work/ml_hw6_pic/eigen/'+filename+'.pkl', 'wb') as f:
        pkl.dump(arr, f)


def load_pkl_data(filename):
    with open('/Users/yen/Work/ml_hw6_pic/eigen/'+filename+'.pkl', 'rb') as f:
        w = pkl.load(f)
    return w


def normal_pick_eigenvector(norm_L, w, n_evector, filename):
    # get eigenvalue and eigenvector
    e_value, e_vector = np.linalg.eig(norm_L)

    D12 = np.diag(1 / np.power(np.sum(w, axis=1), 1 / 2))
    index = np.argsort(e_value)[1:n_evector+1]
    new_space = np.matmul(D12, e_vector[:, index])

    return new_space


def unnormal_pick_eigenvector(unnorm_L, w, n_evector, filename):
    # get eigenvalue and eigenvector
    D_inv = np.diag(1/np.sum(w, axis=1))
    new_matrix = np.matmul(D_inv, unnorm_L)
    e_value, e_vector = np.linalg.eig(new_matrix)

    index = np.argsort(e_value)[1:n_evector+1]
    new_space = e_vector[:, index]

    return new_space



def normalize_cut(w, points, n_cluster, filename):
    norm_L = normalize_L(w)
    new_space = normal_pick_eigenvector(norm_L, w, n_cluster, filename)

    kmeans(10000, n_cluster, new_space, points, 'n')


def ratio_cut(w, points, n_cluster, filename):
    unnorm_L = unnormalize_L(w)
    new_space = unnormal_pick_eigenvector(unnorm_L, w, n_cluster, filename)

    kmeans(10000, n_cluster, new_space, points, 'r')


def draw_result(c, points, times, type, op, center=None):
    plt.figure(num=None, figsize=(6.4, 6.4))
    color_arr = np.array([""] * len(c))

    for i in range(len(c)):
        if c[i] == 0:
            color_arr[i] = 'red'
        elif c[i] == 1:
            color_arr[i] = 'blue'
        elif c[i] == 2:
            color_arr[i] = 'green'
        elif c[i] == 3:
            color_arr[i] = 'yellow'

    plt.gca().invert_yaxis()

    plt.scatter(points[:, 1], points[:, 0], color=color_arr, alpha=0.3, s=7)
    if op == 2:
        draw_center(center)
    plt.savefig('/Users/yen/Work/ml_hw6_pic/s_cluster/' + filename + '_' + str(n_cluster)+'_'+type+'_'+str(times)+'_'+str(op) + '.png')
    plt.show()
    plt.close()


def initial_random(n_cluster, tar_matrix):
    for i in range(n_cluster):
        x_up = np.max(tar_matrix[:, i])
        x_down = np.min(tar_matrix[:, i])
        if i == 0:
            x = np.random.uniform(x_down, x_up, (n_cluster, 1))
        else:
            y = np.random.uniform(x_down, x_up, (n_cluster, 1))
            x = np.concatenate((x, y), axis=1)
    return x


def initial_far(n_cluster, tar_matrix):
    if n_cluster == 2:
        x_up = np.max(tar_matrix[:, 0])
        x_down = np.min(tar_matrix[:, 0])
        y_up = np.max(tar_matrix[:, 1])
        y_down = np.min(tar_matrix[:, 1])

        return np.array([[x_up, y_up],[x_down, y_down]])



def find_label(label, center, tar_matrix, points, times, type):
    for i in range(n_x):

        diff = center - tar_matrix[i, :].reshape(1, -1)
        distance = np.linalg.norm(diff, axis=1)
        label[i] = np.argmin(distance)

    draw_result(label, points, times, type,1)
    draw_result(label, tar_matrix, times, type,2, center)
    return label


def update_center(center, n_cluster, times, tar_matrix, label):
    for i in range(n_cluster):
        center[i, :] = np.mean(tar_matrix[np.where(label == i)[0], :], axis=0)
    return center, times


def draw_center(center):
    plt.scatter(center[:, 1], center[:, 0], color="black", s=7)


def kmeans(n, n_cluster, tar_matrix, points, type):
    center = initial_random(n_cluster, tar_matrix)
    # center = initial_far(n_cluster, tar_matrix)
    label = np.zeros(n)
    times = 0
    while times < 20:
        tag = 0
        print("times", times)
        label = find_label(label, center, tar_matrix, points, times, type)
        # draw_center(center)
        for i in range(n_cluster):
            if len(np.where(label == i)[0]) < 100:
                center = initial_random(n_cluster, tar_matrix)

                times = 0
                tag = -1
                break
        if tag != -1:
            center, times = update_center(center, n_cluster, times, tar_matrix, label)
            times += 1



# start = timeit.default_timer()

filename = 'image2'
img=mpimg.imread(filename+'.png')
img*=255

n_size = img.shape[0]
x = img.reshape(-1, 3)
n_x = x.shape[0]
n_cluster = 2
points = np.array([[i, j] for i in range(n_size) for j in range(n_size)])


w = similarity_matrix(img, 0.001, 0.001)

normalize_cut(w, points, n_cluster, filename)
ratio_cut(w, points, n_cluster, filename)

# stop = timeit.default_timer()
#
# print('Time: ', stop - start)