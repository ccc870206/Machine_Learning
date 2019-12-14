import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
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

#
# def build_design_matrix(n, x):
#     design_matrix = np.empty((1, n))
#     for i in range(n):
#         design_matrix[:, i] = np.power(x, i)
#
#     return design_matrix


def diff_square(x1, x2):
    row = x1.shape[0]
    col = x2.shape[0]
    k = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            k[i][j] = x1[i] - x2[j]
    return k


def rational_quadratic_kernel(x1, x2, alpha):
    k = diff_square(x1, x2)
    k = np.power(np.power(k, 2) + 1, -alpha)

    return k


def func(args):
    k, y, n = args
    # -1/2*ln|c| -1/2*y^t*c^-1*y -N/2*ln(2*pi)
    v = lambda para:-1/2*np.log(np.linalg.det(np.power(np.power(k, 2) + 1, -para)))\
                    -1/2*np.matmul(np.matmul(y.T, np.linalg.inv(np.power(np.power(k, 2) + 1, -para))), y)\
                    -n/2*np.log(2*math.pi)

    return v


def log_likelihood(k, y, n):
    para = 1
    c = np.power(np.power(k, 2) + 1, -para)
    first_term = -1/2*np.log(np.linalg.det(c))
    second_term = -1/2*np.matmul(np.matmul(y.T, np.linalg.inv(c)), y)
    third_term = -n/2*np.log(2*math.pi)
    # third_term = -c.shape[0]/2*math.log(2*math.pi)

    return first_term+second_term+third_term


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
args = (diff_square(x, x), y, x.shape[0])
x0 = np.asarray((1))
res = minimize(func(args), x0, method='SLSQP')


alpha = res.x[0]
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