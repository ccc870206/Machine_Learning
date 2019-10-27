import numpy as np
import math
import matplotlib.pyplot as plt


def generator_gaussian(mean, var):
    return np.random.normal(mean, var)


def input_mean_variance():
    mean = float(input("input mean:"))
    var = float(input("input variance:"))
    return generator_gaussian(mean, var)


def build_design_matrix(n, x):
    design_matrix = np.empty((1, n))
    for i in range(n):
        design_matrix[:, i] = np.power(x, i)

    return design_matrix


def generator_poly_basis(n, a, w):
    x = np.random.uniform(-1, 1, 1)

    A = build_design_matrix(n, x)
    y = np.matmul(A, w)
    e = generator_gaussian(0, a)
    y += e
    # x = -0.64152
    # y = 0.19039
    return np.array([x]), np.array([y])


def input_poly_basis():
    n = int(input("n:"))
    a = int(input("a:"))
    w = input("w:")[1:-1].split(',')
    w = np.array(w, dtype=float)

    return generator_poly_basis(n, a, w)


def new_mean_variance(mean_old, var_old, x_new, n_new):
    mean_new = ((n_new-1)*mean_old + x_new) / n_new
    var_new = var_old**2 + ((x_new-mean_old)*(x_new-mean_new)-(var_old**2))/n_new
    var_new = var_old**2 + (x_new-mean_old)**2/n_new - var_old**2/(n_new-1)
    return mean_new, var_new
# print(input_mean_variance())
# print(input_poly_basis())


def multivariate_gaussian(mean, var, n, x):
    first_term = pow(pow(2*math.pi, 1/2), n/2)*pow(np.linalg.det(var), 2)
    first_term = 1/first_term
    x_minus_m = x - mean
    power_term = -1/2 * np.matmul(np.matmul(np.transpose(x_minus_m),np.linalg.inv(var)),x_minus_m)
    return first_term*pow(math.e, power_term)


def build_identity_matrix(n, lda):
    identity_matrix = np.zeros((n, n))
    for i in range(n):
        identity_matrix[i][i] = lda
    return identity_matrix


# np.linalg.inv
# print(generator_poly_basis(4, 1, [1,2,3,4]))
# print(new_mean_variance(3.408993960833291,0.030383455464755956,0.519242879651157,3))
# print(new_mean_variance(2.445743600439247,1.875958150575018,1.347113997201991,4))
# print(multivariate_gaussian(np.array([1,2]),np.array([[1,0],[0,9]]),2, np.array([2, 2])))


def initial_precision(b, n):

    initial_prior = np.zeros((n, n))
    for i in range(n):
        initial_prior[i][i] = 1/b

    return initial_prior


def output_posterior(poster_mean, poster_variance, predict_mean, predict_var):
    print("Posterior mean:")
    for i in poster_mean:
        print(i[0])
    print()
    print("Posterior variance:")
    for i in poster_variance:
        print(",\t".join(np.asarray(i, dtype=str)))
    print()
    print("Predictive distribution ~ N("+str(predict_mean[0][0])+",", str(predict_var[0][0])+")")
    print('--------------------------------------------------')


def initial_posterior(n, a, b):
    prior = initial_precision(b, n)
    # x = np.array([-0.64152])
    # y = np.array([0.19039])
    x, y = generator_poly_basis(n, a, w)
    print("Add data point ("+str(x[0][0]),","+str(y[0][0])+")")
    mean = 0
    var = prior
    A = build_design_matrix(n, x)

    # Λ = a * A.T * A + bI
    new_var = a * np.matmul(A.transpose(), A) + build_identity_matrix(n, b)
    poster_cov = np.linalg.inv(new_var)

    # M = a * Λ^-1 * A.T * y
    poster_mean = a * np.matmul(np.matmul(np.linalg.inv(new_var), A.transpose()), y).reshape(n, 1)

    # predictive distribution
    predict_mean, predict_var = compute_predictive_distribution(A, a, poster_mean, new_var)

    output_posterior(poster_mean, poster_cov, predict_mean, predict_var)
    return poster_mean, poster_cov, x[0][0], y[0][0]


def gaussian_distribution(mean, std, x):

    if std == 0:
        return 10e-7
    constant = 1/(std * pow(2*math.pi, 1/2))
    power = -(1/2) * pow((x-mean) / std, 2)

    return constant * pow(math.e, power)


def draw_ground_truth(n, w):
    x = np.linspace(-2, 2, 100)
    mean_arr = []
    var_top = []
    var_bot = []

    for i in x:
        A = build_design_matrix(n, i)
        mean = np.matmul(A, w.reshape(n, 1))[0][0]
        var = multivariate_gaussian(np.array([0, 0]), np.array([[1, 0], [0, 1]]), n, np.array([i, i]))
        mean_arr.append(mean)
        var_top.append(mean+var)
        var_bot.append(mean-var)

    plt.plot(x, mean_arr, color="black")
    plt.plot(x, var_top, color="red")
    plt.plot(x, var_bot, color="red")


def compute_predictive_distribution(A, a, poster_mean, new_var):
    # AM
    predict_mean = np.matmul(A, poster_mean)

    # 1/a + A * Λ^-1 * A.T
    predict_var = 1 / a + np.matmul(np.matmul(A, np.linalg.inv(new_var)), A.T)
    return predict_mean, predict_var


def compute_posterior(n, a, prior_mean, prior_cov):
    x, y = generator_poly_basis(n, a, w)
    # x = np.array([0.07122])
    # y = np.array([1.63175])
    A = build_design_matrix(n, x)
    print("Add data point (" + str(x[0][0]), "," + str(y[0][0]) + ")")
    # s = prior_mean^-1
    s = np.linalg.inv(prior_cov)

    # Λ = a * A.T * A + s
    new_var = a * np.matmul(A.transpose(), A) + s
    poster_cov = np.linalg.inv(new_var)

    # M = Λ^-1 * (a* A.T * y + sm)
    poster_mean = np.matmul(np.linalg.inv(new_var), (a * A.transpose() * y + np.matmul(s, prior_mean.reshape(n, 1))))

    # predictive distribution
    predict_mean, predict_var = compute_predictive_distribution(A, a, poster_mean, new_var)

    output_posterior(poster_mean, poster_cov, predict_mean, predict_var)
    return poster_mean, poster_cov, x[0][0], y[0][0]


def draw_mean_var(poster_mean, new_cov, a):
    x = np.linspace(-2, 2, 100)
    mean_arr = []
    var_top = []
    var_bot = []
    for i in x:
        A = build_design_matrix(n, i)
        predict_mean, predict_var = compute_predictive_distribution(A, a, poster_mean, np.linalg.inv(new_cov))
        mean = predict_mean[0][0]
        var = predict_var[0][0]
        mean_arr.append(mean)
        var_top.append(mean + var)
        var_bot.append(mean - var)

    plt.plot(x, mean_arr, color="black")
    plt.plot(x, var_top, color="red")
    plt.plot(x, var_bot, color="red")



def draw_data_point(x, y):

    plt.scatter(x, y)

# def compute_value()


# print(multivariate_gaussian(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]), 2, np.array([[1], [2]])))
# print(multivariate_gaussian(np.array([0, 0]), np.array([[1, 0], [0, 1]]), 2, np.array([1, 2])))

# X = np.random.multivariate_normal(mean=[0,0], cov=[[1, 0], [0,1]], size=1000)
# # print(X)
# plt.scatter(X[:,0], X[:,1])
# plt.show()
b = 1
n = 3
a = 3
w = np.array([1, 2, 3])

fig = plt.subplots(nrows=2, ncols=2)
plt.subplot(221)
plt.title('Ground truth')
draw_ground_truth(n, w)

data_point = []
result = []
predict_range = []
m, c, x, y = initial_posterior(n, a, b)

data_point.append(x)
result.append(y)

for i in range(1000):
    if i == 10:
        plt.subplot(223)
        plt.title('After 10 incomes')
        draw_data_point(data_point, result)
        draw_mean_var(m, c, a)
    if i == 50:
        plt.subplot(224)
        plt.title('After 50 incomes')
        draw_data_point(data_point, result)
        draw_mean_var(m, c, a)
    old_m = m
    m, c, x, y = compute_posterior(n, a, m, c)
    if sum(abs(old_m-m)) < n*10e-6:
        data_point.append(x)
        result.append(y)
        plt.subplot(222)
        plt.title('Predictive result')
        draw_data_point(data_point, result)
        draw_mean_var(m, c, a)
        break

    data_point.append(x)
    result.append(y)
if i == 999:
    data_point.append(x)
    result.append(y)
    plt.subplot(222)
    plt.title('Predictive result')
    draw_data_point(data_point, result)
    draw_mean_var(m, c, a)
plt.show()
print("---------------------------------------")
