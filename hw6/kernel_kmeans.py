import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
import numba as nb
import matplotlib.pyplot as plt
import timeit
import matplotlib.image as mpimg
from scipy import ndimage


def initial_random(n_x, n_cluster):
    c = np.zeros((n_x, n_cluster), dtype=int)
    for i in range(n_x):
        c[i][np.random.randint(0, n_cluster)] = 1
    # print("c", c)
    return c


def initial_far(n_x, size):
    c = np.zeros((n_x, n_cluster), dtype=int)
    # for i in range(n_x):
    #     if i//size > i%size and  i%size > (-i//size+size):
    #         c[i][0] = 1
    #     elif i//size > i%size:
    #         c[i][1] = 1
    #     elif i // size < i % size and i%size > (-i//size+size):
    #         c[i][2] = 1
    #     else:
    #         c[i][3] = 1

    # cluster 2
    for i in range(n_x):
        if i//size > i%size:
            c[i][0] = 1
        elif i//size:
            c[i][1] = 1

    # for i in range(n_x):
    #     if i%size > (-i//size+size):
    #         c[i][0] = 1
    #     elif i//size:
    #         c[i][1] = 1

    return c


def kernel(s1, s2, x1, x2, gamma_s, gamma_c):
    coord = -gamma_s * np.sum(np.power(s2-s1, 2), axis=1)
    color = -gamma_c * np.sum(np.power(x2-x1, 2), axis=1)
    total = coord+color

    return math.e**total


def similarity_matrix(x, position, gamma_s, gamma_c):
    d_coord = -gamma_s * pdist(position, 'sqeuclidean')
    color = x.reshape(-1, 3)
    d_color = -gamma_c * pdist(color, 'sqeuclidean')
    d_total = d_coord + d_color
    return squareform(np.exp(d_total))


def intra_cluster_distance(x, c, size):
    n_x = c.shape[0]
    n_c = c.shape[1]

    total = np.zeros(n_c)

    for j in range(n_c):
        c_list = np.where(c[:, j] == 1)[0]
        x_in_c = x[c_list]
        p_in_c = points[c_list]
        c_sum = c[:, j].sum()
        # print(p_in_c)
        total[j] = (similarity_matrix(x_in_c, p_in_c, gamma_s, gamma_c).sum()+len(c_list))/c_sum/c_sum

    return total


def compute_distance(xi, si, x, c):
    n_c = c.shape[1]
    distance = np.ones(n_c)

    for j in range(n_c):
        sum_c = c[:, j].sum()
        c_list = np.where(c[:, j] == 1)[0]

        c_in_x = x[c_list]
        p_in_x = points[c_list]

        second_term = (kernel(np.array(si), p_in_x, xi, c_in_x, gamma_s, gamma_c).sum()+1)
        second_term = -2*second_term/sum_c
        distance[j] += second_term

    return distance





def draw_result(n_x, c, filename, times):
    plt.figure(num=None, figsize=(6.4, 6.4))
    color_arr = np.array([""] * n_x)
    # print(color_arr.shape)
    for i in range(n_x):
        if np.argmax(c[i]) == 0:
            color_arr[i] = 'red'
        elif np.argmax(c[i]) == 1:
            color_arr[i] = 'blue'
        elif np.argmax(c[i]) == 2:
            color_arr[i] = 'green'
        elif np.argmax(c[i]) == 3:
            color_arr[i] = 'yellow'

    plt.gca().invert_yaxis()

    plt.scatter(points[:, 1], points[:, 0], color=color_arr, alpha=0.3, s=5)
    # plt.savefig('/Users/yen/Work/ml_hw6_pic/'+filename+'_'+str(times)+'.png')
    plt.show()
    plt.close()


gamma_s = 0.0001
gamma_c = 0.0001

image_list = ['image2']
start = timeit.default_timer()
for image in image_list:
    image_name = image
    img = mpimg.imread(image_name + '.png')

    # # show original picture
    # imgplot = plt.imshow(img)
    # plt.show()
    img *= 255

    x = img.reshape(-1, 3)
    n_x = x.shape[0]
    n_size = img.shape[0]
    points = np.array([[i, j] for i in range(n_size) for j in range(n_size)])

    cluster_list = [3]
    for n_cluster in cluster_list:
        c = initial_random(n_x, n_cluster)
        # c = initial_far(n_x, n_size)
        draw_result(n_x, c, image_name, 'default')
        times = 0

        for times in range(20):
            pre_c = c
        # for times in range(10):
            fix_term = intra_cluster_distance(x, c, n_size)
            for i in range(n_x):
                # i = n_x -1
                distance = compute_distance(x[i], (i//n_size, i%n_size), x, c) + fix_term

                c[i, :] = 0
                c[i, np.argmin(distance)] = 1

            draw_result(n_x, c, image_name, times)

        stop = timeit.default_timer()

        print('Time: ', stop - start)