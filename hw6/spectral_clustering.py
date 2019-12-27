import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import timeit
import pickle as pkl
from sklearn.cluster import KMeans
# np.set_printoptions(threshold=np.inf)

def normalize_L(w):
    D = np.diag(np.sum(w, axis=1))
    D12 = np.diag(1/np.power(np.sum(w, axis=1), 1/2))
    L = np.matmul(np.matmul(D12, (D - w)), D12)
    return L


def unnormalize_L(w):
    D = np.diag(np.sum(w, axis=1))
    L = D - w

    return L



def similarity_matrix(x, gamma_s, gamma_c):
    # print("begin")
    coord = np.array([[i, j] for i in range(x.shape[0]) for j in range(x.shape[1])])
    # print("end")
    d_coord = -gamma_s * pdist(coord, 'sqeuclidean')
    # print(squareform(d_coord))
    # print(coord)
    # print(coord.shape)
    # print(squareform(pdist(coord, 'sqeuclidean')))
    color = x.reshape(-1, 3)
    d_color = -gamma_c * pdist(color, 'sqeuclidean')
    # print(squareform(d_color))
    d_total = d_coord + d_color
    # print(d_total)
    return squareform(np.exp(d_total))


def kernel(s1, s2, x1, x2, gamma_s, gamma_c):
    s1 = np.array(s1)
    s2 = np.array(s2)
    coord = -gamma_s * np.sum(np.power(s1-s2, 2))
    color = -gamma_c * np.sum(np.power(x1-x2, 2))
    total = coord+color

    return math.e**total


def intra_cluster_distance(x, size):
    n_x = x.shape[0]


    matrix = np.zeros((n_x, n_x))
    for i in range(n_x):
        matrix[i][i] = 1
        for k in range(i+1, n_x):
            # print(kernel((i // size, i % size), (k // size, k % size), x[i], x[k], 0.01, 0.01))
            value = (kernel((i // size, i % size), (k // size, k % size), x[i], x[k], 0.001, 0.001))
            matrix[i][k] = value
            matrix[k][i] = value
            # print(total[j])
    # print(total)
    return matrix


def save_pkl_data(arr, filename):
    with open('/Users/yen/Work/ml_hw6_pic/eigen/'+filename+'.pkl', 'wb') as f:
        pkl.dump(arr, f)


def load_pkl_data(filename):
    with open('/Users/yen/Work/ml_hw6_pic/eigen/'+filename+'.pkl', 'rb') as f:
        w = pkl.load(f)
    return w


def normal_pick_eigenvector(norm_L, w, n_evector, filename):
    # get eigenvalue and eigenvector
    # e_value, e_vector = np.linalg.eig(norm_L)
    e_value = load_pkl_data(filename+"_n_val")
    e_vector = load_pkl_data(filename+"_n_vec")

    D12 = np.diag(1 / np.power(np.sum(w, axis=1), 1 / 2))
    index = np.argsort(e_value)[1:n_evector+1]
    new_space = np.matmul(D12, e_vector[:, index])

    return new_space


def unnormal_pick_eigenvector(unnorm_L, w, n_evector, filename):
    # # get eigenvalue and eigenvector
    D_inv = np.diag(1/np.sum(w, axis=1))
    # new_matrix = np.matmul(D_inv, unnorm_L)
    # e_value, e_vector = np.linalg.eig(new_matrix)

    # print(new_matrix)


    # save_pkl_data(e_value, filename+"_r_val")
    # save_pkl_data(e_vector, filename+"_r_vec")
    e_value = load_pkl_data(filename+"_r_val")
    e_vector = load_pkl_data(filename+"_r_vec")


    index = np.argsort(e_value)[1:n_evector+1]
    # index = np.argsort(e_value)[[n_evector-1]]
    new_space = e_vector[:, index]
    return new_space
# def kmeans():

def normalize_cut(w, points, n_cluster, filename):
    norm_L = normalize_L(w)

    new_space = normal_pick_eigenvector(norm_L, w, n_cluster, filename)

    kmeans_process(new_space, points)


def ratio_cut(w, points, n_cluster, filename):
    unnorm_L = unnormalize_L(w)

    new_space = unnormal_pick_eigenvector(unnorm_L, w, n_cluster, filename)

    kmeans_process(new_space, points)


def draw_result(c, points):
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
            color_arr[i] = 'cyan'

    plt.gca().invert_yaxis()
    plt.scatter(points[:, 1], points[:, 0], color=color_arr, alpha=0.3, s=5)

    plt.show()
    plt.close()


def kmeans_process(new_space, points):
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(new_space)
    label = kmeans.labels_

    draw_result(label, points)


start = timeit.default_timer()

filename = 'image2'
img=mpimg.imread(filename+'.png')
img*=255
# img=img[:20,:20]
n_size = img.shape[0]
x = img.reshape(-1, 3)
n_x = x.shape[0]
n_cluster = 2
points = np.array([[i, j] for i in range(n_size) for j in range(n_size)])


# print(n_x)

w = similarity_matrix(img, 0.001, 0.001)
# for i in range(n_x):
#     w[i][i] = 1

normalize_cut(w, points, n_cluster, filename)
ratio_cut(w, points, n_cluster, filename)

# w = intra_cluster_distance(x, n_size)
# save_pkl_data(w)
# w = load_pkl_data()

# print(w.shape)

# D = np.zeros((n_x, n_x))
#
# for i in range(n_x):
#     D[i][i] = w[:,i].sum()
"""
D=np.diag(np.sum(w,axis=1))
# print(D.shape)
# print(w)
# unnorm_L = unnormalize_L(w)
norm_L = normalize_L(w)

n_evector = n_cluster
# new_space = unnormal_pick_eigenvector(unnorm_L, w, n_evector)
new_space = normal_pick_eigenvector(norm_L, w, n_evector)
print(new_space)


kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(new_space)
label = kmeans.labels_

draw_result(label, points)

# normal_pick_eigenvector(norm_L, w, n_evector)

# d =
# plt.imshow(w, cmap='gray')
# plt.show()
"""
# e_value, e_vector = np.linalg.eig(L)
# min_index = np.argsort(e_value)
# save_pkl_data(e_value, 'image2_evalue')
# save_pkl_data(e_vector,'image2_evector')


# print(image2_evalue)
# print(image2_evector)

# print(e_value)
# print(min_index)
# G_eval = np.array([[e_value[max_index[1]], 0], [0, e_value[max_index[0]]]])
# G_evec = np.array([e_vector[max_index[1]], e_vector[max_index[0]]])


stop = timeit.default_timer()

print('Time: ', stop - start)