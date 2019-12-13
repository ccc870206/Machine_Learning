import libsvm.python.svmutil as svmutil
import libsvm.python.svm as svm
import numpy as np
import math
import pickle as pkl
import numba as nb

x_train = np.genfromtxt('X_train.csv', delimiter=',')
y_train = np.genfromtxt('Y_train.csv', delimiter=',')
x_test = np.genfromtxt('X_test.csv', delimiter=',')
y_test = np.genfromtxt('Y_test.csv', delimiter=',')
#
#


def convert_format(data):
    x = [{i+1:data[j][i] for i in range(data.shape[1])} for j in range(data.shape[0])]
    return x


def save_pkl_data(x):
    with open('x_train.pkl', 'wb') as f:
        pkl.dump(x, f)


def load_pkl_data():
    with open('x_train.pkl', 'rb') as f:
        x_train = pkl.load(f)

    return x_train


def line_kernel(x1, x2):
    return np.dot(x1,x2)


def poly_kernel(x1, x2, c = 1, d = 2):
    term = c + np.dot(x1, x2)
    return np.power(term, d)


def rbf_kernel(x1, x2, gamma):
    power = -gamma * np.power(x1-x2, 2)
    return math.e**power


# def grid_search(type, option):




# save_pkl_data(convert_format(x_train))
# x_train = load_pkl_data()

x_train = convert_format(x_train)
# x_test = convert_format(x_test)
y_train = list(y_train)
y_test = list(y_test)

prob = svmutil.svm_problem(y_train, x_train)


#		self.c_begin, self.c_end, self.c_step = -5,  15,  2
#		self.g_begin, self.g_end, self.g_step =  3, -15, -2


# para_linear = svmutil.svm_parameter('-t 0 -b 1')
para = '-t 0 -v 10 -c 1 -b 1'
para_linear = svmutil.svm_parameter('-t 0 -v 10 -c 1')
# para_linear = svmutil.svm_parameter('-t 0 -c 1')
# print(para_linear)

model = svmutil.svm_train(prob, para_linear)

# svmutil.svm_save_model("model_file", model)
# p_label, p_acc, p_val = svmutil.svm_predict(y_test, x_test, model)
# print(p_label)

# svmutil.svm_save_model('model_file', model)


#
# y, x = [1,-1], [{1:1, 2:1}, {1:-1,2:-1}]
# prob  = svmutil.svm_problem(y, x)
# param = svmutil.svm_parameter('-t 0 -c 4 -b 1')
# model = svmutil.svm_train(prob, param)
# yt = [1]
# xt = [{1:1, 2:1}]
# p_label, p_acc, p_val = svmutil.svm_predict(yt, xt, model)
# print(p_label)