import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform
import math
import numba as nb
import matplotlib.pyplot as plt
import timeit
import matplotlib.image as mpimg


def initial(n_x, n_cluster):
    c = np.zeros((n_x, n_cluster), dtype=int)
    for i in range(n_x):
        c[i][np.random.randint(0, n_cluster)] = 1
    # print("c", c)
    return c


# @nb.jit()
def kernel(s1, s2, x1, x2, gamma_s, gamma_c):
    s1 = np.array(s1)
    s2 = np.array(s2)
    coord = -gamma_s * np.sum(np.power(s1-s2, 2))
    color = -gamma_c * np.sum(np.power(x1-x2, 2))
    total = coord+color

    return math.e**total


# @nb.jit()
def intra_cluster_distance(x, c, size):
    n_x = c.shape[0]
    n_c = c.shape[1]

    total = np.zeros(n_c)
    for j in range(n_c):
        c_list = np.where(c[:, j] == 1)[0]
        c_sum = c[:, j].sum()
        for i in c_list:
            for k in c_list:
                # print(kernel((i // size, i % size), (k // size, k % size), x[i], x[k], 0.01, 0.01))
                total[j] += (kernel((i // n_x, i % n_x), (k // n_x, k % n_x), x[i], x[k], 0.01, 0.01)/c_sum/c_sum)
                # print(total[j])
    # print(total)
    return total

# @nb.jit()
def compute_distance(xi, si, x, c, size):
    n_c = c.shape[1]

    distance = np.ones(n_c)
    # first_term = kernel(si, si, xi, xi, 0.01, 0.01)
    # distance += first_term
    # print(distance)
    # kernel(si, (0 // n_x, 0 % n_x), xi, x[0], 0.01, 0.01)

    for j in range(n_c):
        second_term = np.array([kernel(si, (i//size,i%size), xi, x[i], 0.01, 0.1) for i in np.where(c[:, j] == 1)[0]])
        # print(second_term.sum())
        second_term = -2*second_term.sum()/c[:, j].sum()
        distance[j] += second_term

    # for j in range(n_c):
    #     second_term = 0
    #     for i in np.where(c[:, j] == 1)[0]:
    #         second_term += kernel(si, (i//size,i%size), xi, x[i], 0.01, 0.01)
    #         # print(second_term.sum())
    #         second_term = -2*second_term/c[:, j].sum()
    #         distance[j] += second_term

    # for j in range(n_c):
    #     third_term = [kernel((i//n_x,i%n_x), (k//size,k%size), x[i], x[k], 0.01, 0.01) \
    #                   for i in np.where(c[:, j] == 1)[0] for k in np.where(c[:, j] == 1)[0]]
    #     print(third_term)
    # # for i in
    # # print(second_term.shape)
    # print(distance)
    return distance


    # for i in range(xi):
    #     kernel()

    # return
start = timeit.default_timer()



np.set_printoptions(threshold=np.inf)
# img = cv2.imread('image2.png')

img=mpimg.imread('image2.png')

imgplot = plt.imshow(img)
img*=255
plt.show()


img = img[:20, :20]

x = img.reshape(-1, 3)
n_size = img.shape[0]
points = np.array([[i, j] for i in range(n_size) for j in range(n_size)])

n_x = x.shape[0]
n_cluster = 2

c = initial(n_x, n_cluster)

for times in range(10):
    fix_term = intra_cluster_distance(x, c, n_size)
    for i in range(n_x):
        distance = compute_distance(x[i], (i//n_size, i%n_size), x, c, n_size) + fix_term
        # print(distance)
        c[i, :] = 0
        c[i, np.argmin(distance)] = 1
        # print(c[i])


    color_arr = np.array([""]*n_x)
    # print(color_arr.shape)
    for i in range(n_x):
        if np.argmax(c[i]) == 0:
            color_arr[i] = 'red'
        elif np.argmax(c[i]) == 1:
            color_arr[i] = 'blue'
        elif np.argmax(c[i]) == 2:
            color_arr[i] = 'green'
        else:
            color_arr[i] = 'yellow'

    plt.scatter(points[:, 0], points[:, 1], color=color_arr, alpha=0.3)
    plt.savefig('/Users/yen/Work/ml_hw6_pic/'+'image2_'+str(times)+'.png')
    plt.show()


# third_term =
# print(intra_cluster_distance(x, c, n_size))



stop = timeit.default_timer()

print('Time: ', stop - start)