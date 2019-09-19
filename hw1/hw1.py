import numpy as np
import matplotlib.pyplot as plt


def load_data():
    input_file = []
    with open('data.txt', 'r') as f:
        for line in f.readlines():
            input_file.append(line.strip().split(','))
    input_file = np.asarray(input_file, dtype=float)
    return input_file, int(input('input n:')), float(input('input lambda:'))
    # return input_file, 3, 10000


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
