import numpy as np
import matplotlib.pyplot as plt
import random


def load_data():
    input_file = []
    with open('data.txt', 'r') as f:
        for line in f.readlines():
            input_file.append(line.strip().split(','))
    input_file = np.asarray(input_file, dtype=float)
    # return input_file, int(input('input n:')), float(input('input lambda:'))
    return input_file, 2, 0


def build_design_matrix(n, x):
    design_matrix = np.empty((x.shape[0], n))
    for i in range(n):
        design_matrix[:, i] = np.power(x, i)

    return design_matrix


def build_identity_matrix(n,lda):
    identity_matrix = np.zeros((n,n))
    for i in range(n):
        identity_matrix[i][i] = lda
    return identity_matrix


def LU_factorization(matrix):
    rank = matrix.shape[0]
    lower = build_identity_matrix(rank, 1)

    for row in range(1,rank):
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

    for row in range(rank-1, -1, -1):
        sum_var = 0
        for col in range(row+1, rank):
            sum_var += solution[col] * matrix[row][col]
        solution[row] = (target[row] - sum_var) / matrix[row][row]

    return solution


def output(para, total_error):
    equation = ''
    for i in range(para.shape[0] - 1,-1,-1):
        if i == 0:
            equation += (str(para[i][0]))
        else:
            equation += (str(para[i][0]) + ' x^' + str(i) + ' + ')

    print('Fitting line:', equation)
    print('Total error:', total_error)


def initial_matrix(n):
    init_x = np.zeros((n,1))
    for i in range(n):
        init_x[i][0] = random.uniform(0,5)

    return init_x


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


def compute_step(A, b, x):
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
    # inverse = np.empty((n, n))
    # for i in range(f.shape[1]):
    #     y = forward_substitution(L, gradient[:,i])
    #     x = backward_substitution(U, y)
    #     inverse[:, i] = np.transpose(x)

    # print(inverse)

    return x - inverse
    # print(H)
    # print(gradient)

def draw_equation(para):
    x = np.linspace(-5, 5, 100)
    y = np.zeros(100)
    print('x', x.shape)
    #
    # print((para[1][0]*np.power(x,1) + y).shape)
    # print((para[1][0]*np.power(x,1) + y).shape)
    for i in range(para.shape[0]):
        y += (para[i][0] * np.power(x, i))

    plt.plot(x, y, color="black")
    # plt.show()


def LSE():
    data, n, lda = load_data()
    A = build_design_matrix(n, data[:, 0])
    b = data[:, 1]
    # print(data)
    At = np.transpose(A)
    AtxA = np.matmul(At, A) + build_identity_matrix(n, lda)
    Atxb = np.matmul(At, b)

    L, U = LU_factorization(AtxA)
    y = forward_substitution(L, Atxb)
    x = backward_substitution(U, y)
    # print(x)

    F = np.matmul(A, x)
    plt.scatter(data[:,0],F)

    error = np.sum(np.power(F[:, 0] - b, 2))
    # print('error:',error)
    print('LSE:')
    output(x, error)

    plt.scatter(data[:, 0],b)

    plt.show()


data, n, lda = load_data()
x = initial_matrix(n)
# print(x)
A = build_design_matrix(n, data[:, 0])
b = data[:, 1]
result = compute_step(A, b, x)

print(result)
draw_equation(result)

F = np.matmul(A, result)
#
#
error = np.sum(np.power(F[:, 0] - b, 2))
print('Newton\'s Method::')
output(result, error)

plt.scatter(data[:, 0], b)

plt.show()