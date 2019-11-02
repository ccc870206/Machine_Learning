import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def generator_gaussian(mean, var):
    return np.random.normal(mean, var**(1/2))


def logistic_function(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i][j] = 1 / (1 + math.e ** (-matrix[i][j]))

    return matrix


def input_data():
    n = int(input("n:"))
    mx1 = float(input("mx1:"))
    my1 = float(input("my1:"))
    mx2 = float(input("mx2:"))
    my2 = float(input("my2:"))

    vx1 = float(input("vx1:"))
    vy1 = float(input("vy1:"))
    vx2 = float(input("vx2:"))
    vy2 = float(input("vy2:"))

    return n, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2


def initial_parameter():
    n, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2 = input_data()

    D1 = np.empty((n, 2))
    D2 = np.empty((n, 2))

    for i in range(n):
        D1[i][0] = generator_gaussian(mx1, vx1)
        D1[i][1] = generator_gaussian(my1, vy1)
        D2[i][0] = generator_gaussian(mx2, vx2)
        D2[i][1] = generator_gaussian(my2, vy2)

    Dx = np.append(D1[:, 0], D2[:, 0], axis=0)
    Dy = np.append(D1[:, 1], D2[:, 1], axis=0)
    Dclass = np.append(np.zeros(n), np.ones(n), axis=0)
    phi = build_design_matrix(n, Dx, Dy)

    return n, Dx, Dy, Dclass, phi


def compute_D(matrix):
    D = np.zeros((matrix.shape[1], matrix.shape[1]),dtype="float64")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            D[j][j] = math.e**(-matrix[i][j])/(1 + math.e**(-matrix[i][j]))/(1 + math.e ** (-matrix[i][j]))

    return D


def build_design_matrix(n, x, y):
    one_arr = np.ones(2*n)

    design_matrix = np.append(one_arr, x, axis=0)
    design_matrix = np.append(design_matrix, y, axis=0)

    return design_matrix.reshape((3, -1))


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


def predict(phi, w):
    result = np.matmul(phi.T, w)

    prediction = []
    for i in result:
        prediction.append(1) if i > 1 / 2 else prediction.append(0)

    return prediction


def gradient_descent(phi, Dclass):
    w = build_random_matrix(3)

    while True:
        gradient = compute_gradient(phi, Dclass, w)
        w -= gradient

        if abs(np.sum(gradient)) < 3 * 10e-3:
            break

    prediction = predict(phi, w)

    print("Gradient descent:")
    print()
    print("w:")
    [print(i[0]) for i in w]
    print()
    output_confusion_matrix(Dclass, prediction)

    return prediction


def newton(phi, Dclass):
    lr = 0.005
    nt_times = 0
    w = build_random_matrix(3)

    while True:
        D = compute_D(np.matmul(w.T, phi))
        H = np.matmul(np.matmul(phi, D), phi.T)
        gradient = compute_gradient(phi, Dclass, w)

        if np.linalg.det(H) == 0:
            if nt_times > 1000 and abs(np.sum(gradient)) < 3 * 10e-3:
                break
            w -= lr * gradient
        else:
            update = np.matmul(np.linalg.inv(H), gradient)
            if nt_times > 1000 and abs(np.sum(update)) < 3 * 10e-3:
                break
            w -= lr * update
        nt_times += 1

    prediction = predict(phi, w)

    print("Newton's method:")
    print()
    print("w:")
    [print(i[0]) for i in w]
    print()

    output_confusion_matrix(Dclass, prediction)

    return prediction


def draw_result(gt, gd, nt):
    color_gt = convert_color(gt)
    color_gd = convert_color(gd)
    color_nt = convert_color(nt)

    plt.subplot(131)
    plt.title('Ground Truth')
    plt.scatter(Dx, Dy, color=color_gt)

    plt.subplot(132)
    plt.title('Gradient Descent')
    plt.scatter(Dx, Dy, color=color_gd)
    plt.show()

    plt.subplot(133)
    plt.title('Newton\'s method')
    plt.scatter(Dx, Dy, color=color_nt)


if __name__ == "__main__":
    n, Dx, Dy, Dclass, phi = initial_parameter()
    prediction_nt = newton(phi, Dclass)
    prediction_gd = gradient_descent(phi, Dclass)
    draw_result(Dclass, prediction_gd, prediction_nt)