import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform
import math
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
img = cv2.imread('image1.png')
# img = cv2.imread('image2.png')

# print(img.shape)
# cv2.imshow('My Image', img)


def whole_kernel(x, gamma_s, gamma_c):
    # print(x.shape)
    coord = np.array([[i, j] for i in range(x.shape[0]) for j in range(x.shape[1])])
    # coord = coord / n_size
    d_coord = -gamma_s*pdist(coord, 'sqeuclidean')
    # print(squareform(d_coord))
    # print(coord)
    # print(coord.shape)
    # print(squareform(pdist(coord, 'sqeuclidean')))
    color = x.reshape(-1, 3)
    d_color = -gamma_c*pdist(color, 'sqeuclidean')
    # print(squareform(d_color))
    d_total = d_coord+d_color
    return squareform(np.exp(d_total))
    # print(d_color)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         print()


# def vector_kernel(s1, s2, x1, x2, gamma_s, gamma_c):
#     print("s1,s2",s1,s2)
#     print("x1,x2",x1,x2)
#     coord = -gamma_s * np.sum(np.power(s1-s2, 2))
#     color = -gamma_c * np.sum(np.power(x1-x2, 2))
#     total = coord+color
#     print(coord, color)
#     return math.e**total

# [[3600 7921]
#  [4624 2116]]

def KernelTrick(gram_m, c):
    """
    calculate Euclidean distance in feature space by kernel trick
    parameters:
        gram_m : gram matrix K(x, x)
        c : cluster vector c(i,k) = 1 if x(i) belong k clustering
    return:
        w : n*k
    """
    return (
        np.matmul(
        gram_m * np.eye(gram_m.shape[0]),
        np.ones((gram_m.shape[0], c.shape[1]))) - 2*( np.matmul(gram_m, c) / np.sum(c, axis=0) ) + (np.matmul(np.ones((gram_m.shape[0], c.shape[1])),      np.matmul(np.matmul(c.T, gram_m), c)*np.eye(c.shape[1])) / (np.sum(c,axis=0)**2))
    )



def vector_kernel(s1, s2, x1, x2, gamma_s, gamma_c):
    # print("s1,s2",s1,s2)
    # print("x1,x2",x1,x2)
    print("power:", np.power(s1-s2, 2))
    coord = -gamma_s * np.sum(np.power(s1-s2, 2), axis=1)
    color = -gamma_c * np.sum(np.power(x1-x2, 2), axis=1)
    total = coord+color
    # print(coord, color)
    return math.e**total


def initialize(n_size ,n_cluster):
    mean_coord = np.random.randint(0, n_size, (n_cluster, 2))
    mean_color = np.random.randint(0, 255, (n_cluster, 3))
    # mean_color = mean_color/255
    # mean_coord = mean_coord/n_size

    return mean_coord, mean_color


def initial(n_x, n_cluster):
    c = np.zeros((n_x, n_cluster), dtype=int)
    for i in range(n_x):
        c[i][np.random.randint(0, n_cluster)] = 1
    print("c", c)
    return c


x = img
# x = np.array([[[ 29,177,109],[203,197,18]],[[1,6,3],[3,4,2]]])
# x = x/255
# x = np.array([[[1,2,3],[3,4,2]],[[1,6,3],[3,4,2]]])


n_size = x.shape[0]
n_cluster = 2
mean_coord, mean_color = initialize(n_size, n_cluster)

points = np.array([[i, j] for i in range(n_size) for j in range(n_size)])

print("mean_coord", mean_coord)
x = x.reshape(-1, 3)

n_x = x.shape[0]
# print("n_size", n_size)
# print("n_x",n_x)

# c = np.zeros((n_x, n_cluster),dtype=int)

# c = np.zeros(n_x, dtype=int)

sum = 0
# c[:5000,0] = 1
# c[5000:,1] = 1


# print(c)
c = initial(n_x, n_cluster)


color_arr = np.array([""]*n_x)
print(color_arr.shape)
for i in range(n_x):
    # print(c[i])
    if np.argmax(c[i]) == 0:
        color_arr[i] = 'red'
    else:
        color_arr[i] = 'blue'
plt.scatter(points[:, 0], points[:, 1], color=color_arr, alpha=0.3)
plt.show()
# for times in range(5):
for times in range(10):
    n_c = np.zeros(n_cluster)
    # for i in range(n_x):
    #     x_coord = np.array([i//n_size, i%n_size])
    #     # print(vector_kernel(x_coord, x_coord, x[i], x[i], 0.001, 0.001))
    #     # print(vector_kernel(x_coord, mean_coord, x[i], mean_color, 0.001, 0.001))
    #     # print("kernel", KernelTrick(whole_kernel(img, 1, 1), c))
    w = KernelTrick(whole_kernel(img, 1, 1), c)
    update_c = np.zeros(w.shape)
    update_c[np.arange(w.shape[0]), np.argmin(w, axis=1)] = 1

    delta_c = np.count_nonzero(np.abs(update_c - c))

    c = update_c
    # distance = vector_kernel(x_coord, x_coord, x[i], x[i], 0.001, 0.001) \
    #            + vector_kernel(mean_coord, mean_coord, mean_color, mean_color, 0.001, 0.001) \
    #            - 2*vector_kernel(x_coord, mean_coord, x[i], mean_color, 0.001, 0.001)
    # print("distance", distance)
    # print(np.argmin(distance))
    """if distance.sum() == 0:                     """
    """    c[i] = np.random.randint(0, n_cluster)  """
    """    # print("up",c[i])                      """
    """    sum += 1                                """
    """else:                                       """
    """    # print(np.argmin(distance))            """
    """    c[i] = np.argmin(distance)              """
    """    # print("down",c[i])                    """
    """n_c[c[i]] += 1                              """
    # c[i] = np.argmin(distance)
    # n_c[c[i]] += 1
        # print(c[i])
    """
    mean_coord_new = np.zeros((n_cluster, 2))
    mean_color_new = np.zeros((n_cluster, 3))
    for i in range(n_x):
        mean_coord_new[c[i], :] += np.array([i//n_size, i%n_size])
        mean_color_new[c[i], :] += x[i]

    # n_c[0] = 2
    # n_c[1] = 1
    # print(mean_coord_new.shape)
    # print(n_c.shape)
    # print(mean_coord_new)
    # print(n_c)
    for i in range(n_cluster):
        mean_coord[i, :] = mean_coord_new[i, :] / n_c[i] if n_c[i] != 0 else np.zeros((1,2))
        # print(mean_coord_new[i, :])
        mean_color[i, :] = mean_color_new[i, :] / n_c[i] if n_c[i] != 0 else np.zeros((1,3))

    print(times+1, "round")
    print("mean_coord", mean_coord)
    print("mean_coord", mean_coord.shape)
    print("mean_color", mean_color.shape)

    """
    if times % 2 == 0:
        color_arr = np.array([""]*n_x)
        print(color_arr.shape)
        for i in range(n_x):
            if np.argmax(c[i]) == 0:
                color_arr[i] = 'red'
            else:
                color_arr[i] = 'blue'
        plt.scatter(points[:, 0], points[:, 1], color=color_arr, alpha=0.3)
        plt.show()

# 
# print(mean_coord_new)
# print(mean_color_new)




# print(x)
# print(x.shape)
# print(squareform(pdist(x, 'sqeuclidean')))
# k = whole_kernel(img, 1,1)
# (10000, 10000)
# print(k.shape)

x_new = x.reshape(-1, 3)
# print(x_new)
# print(vector_kernel(np.array([0, 0]), np.array([1, 1]), x_new[0], x_new[3], 1, 1))

# (5000, 784)
