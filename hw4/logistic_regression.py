import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
import sys

# if not sys.warnoptions:
#     warnings.simplefilter("ignore")


def generator_gaussian(mean, var):
    return np.random.normal(mean, var**(1/2))


def logistic_function(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i][j] = 1 / (1 + math.e ** (-matrix[i][j]))

    return matrix


def compute_D(matrix):
    print("m")
    print(matrix)
    D = np.zeros((matrix.shape[1], matrix.shape[1]),dtype="float64")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # log_value = -matrix[i][j]-2*math.log()
            D[j][j] = math.e**(-matrix[i][j])/(1 + math.e**(-matrix[i][j]))/(1 + math.e ** (-matrix[i][j]))
            # print("son", matrix[i][j], math.e**(-matrix[i][j]))
            # print("mom", matrix[i][j], (1 + math.e**(-matrix[i][j])))

    return D



def build_design_matrix(n, x):
    design_matrix = np.empty((x.shape[0], n))
    for i in range(n):
        design_matrix[:, i] = np.power(x, i)
    design_matrix = design_matrix.T
    return design_matrix


def build_random_matrix(n):
    return np.random.uniform(-5, 5, (n, 1))


def compute_gradient(phi, b, w):
    # At * (logistic - b)
    logistic = logistic_function(np.matmul(w.T, phi))
    second_term = logistic - b

    return np.matmul(phi, second_term.T)


def convert_color(vector):
    color = []
    for i in vector:
        color.append('red')if i == 0 else color.append('blue')
    return color


def build_confusion_matrix(gt, pred):
    # gt pred
    c11 = 0
    c12 = 0
    c21 = 0
    c22 = 0
    for i in range(gt.shape[0]):
        if gt[i] == 0:
            if pred[i] == 0:
                c11 += 1
            else:
                c12 += 1
        else:
            if pred[i] == 0:
                c21 += 1
            else:
                c22 += 1
    return c11, c12, c21, c22



def output_confusion_matrix(gt, pred):
    c11, c12, c21, c22 = build_confusion_matrix(gt, pred)
    sensitivity = c11/(c11+c12)
    specificity = c22/(c21+c22)

    print("Confusion Matrix:")
    print("\t\t\tPredict cluster 1","Predict cluster 2")
    print("Is cluster 1\t\t\t", c11, "\t", c12)
    print("Is cluster 2\t\t\t", c21, " \t", c22)
    print()
    print("Sensitivity (Successfully predict cluster 1):", sensitivity)
    print("Specificity (Successfully predict cluster 2):", specificity)




n = 50
# mx1 = 1
# my1 = 1
# mx2 = 10
# my2 = 10
#
# vx1 = 2
# vy1 = 2
# vx2 = 2
# vy2 = 2

mx1 = 1
my1 = 1
mx2 = 3
my2 = 3

vx1 = 2
vy1 = 2
vx2 = 4
vy2 = 4


D1 = np.empty((n, 2))
D2 = np.empty((n, 2))

for i in range(n):
    D1[i][0] = generator_gaussian(mx1, vx1)
    D1[i][1] = generator_gaussian(my1, vy1)
    D2[i][0] = generator_gaussian(mx2, vx2)
    D2[i][1] = generator_gaussian(my2, vy2)

w = build_random_matrix(3)
Dx = np.append(D1[:, 0], D2[:, 0], axis=0)
Dy = np.append(D1[:, 1], D2[:, 1], axis=0)
Dclass = np.append(np.zeros(n), np.ones(n), axis=0)
# print(D1[:, 0])
# print(D2[:, 0])
phi = build_design_matrix(3, Dx)






# print(A)
# A1 = build_design_matrix(3, D1[:, 0])
# A2 = build_design_matrix(3, D2[:, 0])

# print(A1)

# print("Dy", Dy.shape)
# print("Dclass:", Dclass.shape, Dclass)
# w

# i = 0
# for i in range(50):
#     D = compute_D(np.matmul(w.T, phi))
#     print(D)
#     H = np.matmul(np.matmul(phi, D), phi.T)
#     gradient = compute_gradient(phi, Dclass, w)
#     # print(H)
#     if np.linalg.det(H) == 0:
#         w -= gradient
#     else:
#         w -= np.matmul(np.linalg.inv(H), gradient)
#     print(w)
#
# result = np.matmul(phi.T, w)
#
# prediction = []
# for i in result:
#     prediction.append(1) if i > 1/2 else prediction.append(0)
# print(prediction)


## gradient descent
# while(True):
for i in range(1000):
    gradient = compute_gradient(phi, Dclass, w)
    w -= gradient
    print(w)
    if abs(np.sum(gradient)) < 3*10e-2:
        break

result = np.matmul(phi.T, w)

prediction = []
for i in result:
    prediction.append(1) if i >= 1/2 else prediction.append(0)

output_confusion_matrix(Dclass, prediction)


color_pre = convert_color(prediction)
color_gt = convert_color(Dclass)
# print(convert_color(prediction))


plt.subplot(131)
plt.title('Ground Truth')
plt.scatter(Dx, Dy, color=color_gt)

plt.subplot(132)
plt.title('Gradient Descent')
plt.scatter(Dx, Dy, color=color_pre)
plt.show()



# print(compute_gradient(phi, Dclass, w))

# gradient = compute_gradient(A, A.T, Dy, w).reshape(3, 1)
# print(w - gradient)
# w = w - gradient
# gradient = compute_gradient(A, A.T, Dy, w).reshape(3, 1)
# print(w-gradient)
# print(logistic_function(0))