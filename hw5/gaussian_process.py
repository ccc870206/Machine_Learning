import numpy as np
import matplotlib.pyplot as plt
import math
# from  scipy.stats import multivariate_normal as multi_gaussian


def generator_gaussian(mean, var):
    return np.random.normal(mean, var**(1/2))

#
# def gaussian_distribution(mean, std, x):
#     if std == 0:
#         return 10e-7
#     constant = 1/(std * pow(2*math.pi, 1/2))
#     power = -(1/2) * pow((x-mean) / std, 2)
#
#     return constant * pow(math.e, power)


def build_design_matrix(n, x):
    design_matrix = np.empty((1, n))
    for i in range(n):
        design_matrix[:, i] = np.power(x, i)

    return design_matrix


def rational_quadratic_kernel(x1, x2, alpha):
    row = x1.shape[0]
    col = x2.shape[0]
    k = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            k[i][j] = x1[i] - x2[j]
    k = np.power(np.power(k, 2) + 1, -alpha)

    return k


f = open('input.data', 'r')
data = np.array([])
for i in f.readlines():
    data = np.append(data, float(i.split()[0]))
    data = np.append(data, float(i.split()[1]))

data = data.reshape((34, 2))
x = data[:, 0]
y = data[:, 1]

noise = generator_gaussian(0, 1/5)

test = np.linspace(-60, 60, num=1000)
alpha = 1
c = rational_quadratic_kernel(x, x, alpha)
k_x_test = rational_quadratic_kernel(x, test, alpha)
k_x_test_t = k_x_test.T
k_test_test = rational_quadratic_kernel(test, test, alpha)

mean = np.matmul(np.matmul(k_x_test_t, np.linalg.inv(c)), y)
var = (k_test_test + noise) - np.matmul(np.matmul(k_x_test_t, np.linalg.inv(c)), k_x_test)
var = np.diag(var)


plt.scatter(data[:, 0], data[:, 1])
plt.plot(test, mean, color="black")
plt.fill_between(test, mean-2*var, mean+2*var, facecolor="red", alpha=0.3)

plt.show()