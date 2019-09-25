import numpy as np
import matplotlib.pyplot as plt
import random


def load_data():
    input_file = []
    with open('data.txt', 'r') as f:
        for line in f.readlines():
            input_file.append(line.strip().split(','))
    input_file = np.asarray(input_file, dtype=float)
    return input_file, int(input('input n:')), float(input('input lambda:'))


def build_design_matrix(n, x):
    design_matrix = np.empty((x.shape[0], n))
    for i in range(n):
        design_matrix[:, i] = np.power(x, i)

    return design_matrix


def build_identity_matrix(n, lda):
    identity_matrix = np.zeros((n, n))
    for i in range(n):
        identity_matrix[i][i] = lda
    return identity_matrix


def build_random_matrix(n):
    init_x = np.zeros((n, 1))
    for i in range(n):
        init_x[i][0] = random.uniform(0, 5)

    return init_x


def LU_factorization(matrix):
    rank = matrix.shape[0]
    lower = build_identity_matrix(rank, 1)

    for row in range(1, rank):
        for col in range(row):
            e = matrix[row][col] / matrix[col][col]
            lower[row][col] = e
            matrix[row, :] -= e * matrix[col, :]
    matrix = np.round(matrix, 5)
    return lower, matrix


def forward_substitution(matrix, target):
    rank = matrix.shape[0]
    solution = np.empty((rank, 1))
    for row in range(rank):
        sum_var = 0
        for col in range(row):
            sum_var += solution[col] * matrix[row][col]
        solution[row] = (target[row] - sum_var)

    return solution


def backward_substitution(matrix, target):
    rank = matrix.shape[0]
    solution = np.empty((rank, 1))

    for row in range(rank - 1, -1, -1):
        sum_var = 0
        for col in range(row + 1, rank):
            sum_var += solution[col] * matrix[row][col]
        solution[row] = (target[row] - sum_var) / matrix[row][row]

    return solution


def compute_LSE(A, b):
    At = np.transpose(A)
    AtxA = np.matmul(At, A) + build_identity_matrix(n, lda)
    Atxb = np.matmul(At, b)

    L, U = LU_factorization(AtxA)
    y = forward_substitution(L, Atxb)
    x = backward_substitution(U, y)
    return x


def compute_f(A, At, x, xt, b, bt):
    # xt * At * A * x
    first = np.matmul(np.matmul(np.matmul(xt, At), A), x)
    # 2 * xt * At * b
    second = 2 * np.matmul(np.matmul(xt, At), b)
    # bt * b
    third = np.matmul(bt, b)

    return first - second + third


def compute_gradient(A, At, b, x):
    # 2 * At * A * x
    first = 2 * np.matmul(np.matmul(At, A), x)

    # 2 * At * b
    second = 2 * np.matmul(At, b)

    return first[:, 0] - second


def compute_H(A, At):
    return 2 * np.matmul(At, A)


def compute_newton(A, b, x):
    xt = np.transpose(x)
    At = np.transpose(A)
    bt = np.transpose(b)
    f = compute_f(A, At, x, xt, b, bt)
    gradient = compute_gradient(A, At, b, x)
    # print('gradient', gradient)
    H = compute_H(A, At)
    L, U = LU_factorization(H)

    y = forward_substitution(L, gradient)
    inverse = backward_substitution(U, y)

    return x - inverse


def draw_equation(para):
    x = np.linspace(-5, 5, 100)
    y = np.zeros(100)

    for i in range(para.shape[0]):
        y += (para[i][0] * np.power(x, i))

    plt.plot(x, y, color="black")


def draw_groundtruth(x, y):
    plt.scatter(x, y)


def print_result(para, total_error):
    equation = ''
    for i in range(para.shape[0] - 1, -1, -1):
        if i == 0:
            equation += (str(para[i][0]))
        else:
            equation += (str(para[i][0]) + ' x^' + str(i) + ' + ')

    print('Fitting line:', equation)
    print('Total error:', total_error)


def output(A, b, data_x, para):
    predict_y = np.matmul(A, para)
    error = np.sum(np.power(predict_y[:, 0] - b, 2))

    draw_groundtruth(data_x, b)
    draw_equation(para)
    print_result(para, error)


def LSE(A, b):
    result = compute_LSE(A, b)

    print('LSE:')
    output(A, b, data[:, 0], result)


def Newton_method(A, b):
    x = build_random_matrix(n)

    result = compute_newton(A, b, x)

    print('Newton\'s Method:')
    output(A, b, data[:, 0], result)


if __name__ == "__main__":
    data, n, lda = load_data()
    A = build_design_matrix(n, data[:, 0])
    b = data[:, 1]

    fig = plt.subplots(nrows=2, ncols=1)
    plt.subplot(211)
    plt.title('LSE')
    LSE(A, b)

    plt.subplot(212)
    plt.title('Newton')
    Newton_method(A, b)
    plt.show()
