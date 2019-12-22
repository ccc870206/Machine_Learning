import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform
import math

img = cv2.imread('image1.png')
# img = cv2.imread('image2.png')

# print(img.shape)
# cv2.imshow('My Image', img)


def whole_kernel(x, gamma_s, gamma_c):
    # print(x.shape)
    coord = np.array([[i, j] for i in range(x.shape[0]) for j in range(x.shape[1])])
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


def vector_kernel(s1, s2, x1, x2, gamma_s, gamma_c):
    # print("s1,s2",s1,s2)
    # print("x1,x2",x1,x2)
    coord = -gamma_s * np.sum(np.power(s1-s2, 2), axis=1)
    color = -gamma_c * np.sum(np.power(x1-x2, 2), axis=1)
    total = coord+color
    # print(coord, color)
    return math.e**total



def initialize(n_size ,n_cluster):
    mean_coord = np.random.randint(0, n_size, (n_cluster, 2))
    mean_color = np.random.randint(0, 255, (n_cluster, 3))
    mean_color = mean_color/255


    return mean_coord, mean_color

x = img
# x = np.array([[[ 29,177,109],[203,197,18]],[[1,6,3],[3,4,2]]])
x = x/255
# x = np.array([[[1,2,3],[3,4,2]],[[1,6,3],[3,4,2]]])
n_size = x.shape[0]
n_cluster = 2
mean_coord, mean_color = initialize(n_size, n_cluster)

x = x.reshape(-1, 3)

n_x = x.shape[0]
# print("n_size", n_size)
# print("n_x",n_x)
c = np.zeros(n_x)

sum = 0

for i in range(n_x):
    x_coord = np.array([i//n_size, i%n_size])
    distance = vector_kernel(x_coord, mean_coord, x[i], mean_color, 1, 1)
    # print("distance", distance)
    # print(np.argmin(distance))
    if distance.sum() == 0:
        c[i] = np.random.randint(0, n_cluster)
        sum += 1
    else:
        c[i] = np.argmin(distance)
    # print(c[i])

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
