import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize


def generator_gaussian(mean, var):
    return np.random.normal(mean, var**(1/2))


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
    # 1/2*ln(c) -1/2*y^t*c^-1*y -N/2*ln(2*pi)
    v = lambda para:-1/2*-para*np.log(np.linalg.det(np.power(k, 2) + 1)+10e-7)\
                    -1/2*np.matmul(np.matmul(y.T, np.linalg.inv(np.power(np.power(k, 2) + 1, -para))), y)\
                    -n/2*np.log(2*math.pi)
    return v


def con():
    cons = ({'type': 'ineq', 'fun': lambda para: para})
    return cons


def load_data():
    f = open('input.data', 'r')
    data = np.array([])
    for i in f.readlines():
        data = np.append(data, float(i.split()[0]))
        data = np.append(data, float(i.split()[1]))

    data = data.reshape((34, 2))
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def optimize_process(x, y):
    args = (diff_square(x, x), y, x.shape[0])
    cons = con()
    x0 = np.asarray((10e-7))
    res = minimize(func(args), x0, method='SLSQP', constraints=cons)

    return res.x[0]


def predict_process(x, alpha, beta_inv):
    c = rational_quadratic_kernel(x, x, alpha)
    k_x_test = rational_quadratic_kernel(x, test, alpha)
    k_x_test_t = k_x_test.T
    k_test_test = rational_quadratic_kernel(test, test, alpha)

    # mean = k_x_test_t * c^-1 * y
    mean = np.matmul(np.matmul(k_x_test_t, np.linalg.inv(c)), y)
    # var = (k_test_test + beta^-1) - k_x_test_t * c^-1 *
    var = (k_test_test + beta_inv) - np.matmul(np.matmul(k_x_test_t, np.linalg.inv(c)), k_x_test)
    var = np.diag(var)

    return mean, var


def gaussian_process(x, y, beta_inv):
    alpha = optimize_process(x, y)
    mean, var = predict_process(x, alpha, beta_inv)
    return mean, var


def visualize_result(x, y, mean, var):
    plt.scatter(x, y)
    plt.plot(test, mean, color="black")
    plt.fill_between(test, mean - 2 * var, mean + 2 * var, facecolor="red", alpha=0.3)

    plt.show()


if __name__ == '__main__':
    x, y = load_data()
    beta_inv = 1/5
    noise = generator_gaussian(0, beta_inv)
    test = np.linspace(-60, 60, num=100)

    mean, var = gaussian_process(x, y, beta_inv)

    visualize_result(x, y, mean, var)

