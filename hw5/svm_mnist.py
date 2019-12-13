import libsvm.python.svmutil as svmutil
import libsvm.python.svm as svm
import libsvm.python.grid as grid
import numpy as np
import math
import pickle as pkl
import numba as nb
from scipy.spatial.distance import pdist, squareform
import timeit

x_train_origin = np.genfromtxt('X_train.csv', delimiter=',')
y_train = np.genfromtxt('Y_train.csv', delimiter=',')
x_test_origin = np.genfromtxt('X_test.csv', delimiter=',')
y_test = np.genfromtxt('Y_test.csv', delimiter=',')


def convert_format(data):
    x = [{i+1:data[j][i] for i in range(data.shape[1])} for j in range(data.shape[0])]
    return x


def line_kernel(x):
    return np.matmul(x, x.T)


def poly_kernel(x, c = 1, d = 2):
    term = c + np.matmul(x)
    return np.power(term, d)


def rbf_kernel(x, gamma):
    # power = -gamma * np.power(x1-x2, 2)
    # return np.exp(power)

    return squareform(np.exp(-gamma * pdist(x, 'sqeuclidean')))


def grid_search(prob, type, c, g = [], r = [], d = []):
    if type == 0:
        option = '-q -t 0 -v 10 '
        acc = np.array([])
        f_linear = open('linear_op.csv', 'w')
        f_linear.write('c_op,-c,acc\n')
        for i in c:
            para = option + '-c ' + str(2**i)
            model = svmutil.svm_train(prob, para)
            acc = np.append(acc, model)
            f_linear.write(str(i)+','+str(2**i)+','+str(model)+'\n')
        print(acc)
        print(c[np.argmax(acc)])
        best_para = ' -c ' + str(2**c[np.argmax(acc)])

        return best_para
    elif type == 1:
        option = '-q -t 1 -v 10 '
        acc = np.array([])
        f_poly = open('poly_op.csv', 'w')
        f_poly.write('c_op,-c,-g,-r,-d,acc\n')
        for i in c:
            for j in g:
                for k in r:
                    for l in d:
                        para = option + '-c ' + str(2**i) + ' -g ' + str(j) + \
                               ' -r ' + str(k) + ' -d ' + str(l)
                        model = svmutil.svm_train(prob, para)
                        acc = np.append(acc, model)
                        f_poly.write(str(i)+','+str(2**i)+','+str(j)+','+str(k)+','+str(l)+','+str(model)+'\n')
        acc = acc.reshape(len(c), len(g), len(r), len(d))
        idx = np.unravel_index(np.argmax(acc), acc.shape)

        print(acc)
        print(idx)
        best_para = ' -c ' + str(2**c[idx[0]]) + ' -g ' + str(g[idx[1]]) + ' -r ' + str(r[idx[2]]) + ' -d ' + str(d[idx[3]])

        return best_para

    elif type == 2:
        option = '-q -t 2 -v 10 '
        acc = np.array([])
        f_rbf = open('rbf_op.csv', 'w')
        f_rbf.write('c_op,-c,-g,acc\n')
        for i in c:
            for j in g:
                para = option + '-c ' + str(2**i) + ' -g ' + str(j)
                model = svmutil.svm_train(prob, para)
                print(model)
                acc = np.append(acc, model)
                f_rbf.write(str(i)+','+str(2**i)+','+str(j)+','+str(model)+'\n')
        acc = acc.reshape((len(c), len(g)))
        idx = np.unravel_index(np.argmax(acc), acc.shape)

        print(acc)
        print(idx)
        best_para = ' -c ' + str(2**c[idx[0]]) + ' -g ' + str(g[idx[1]])

        return best_para

    elif type == 4:
        option = '-q -t 4 -v 10 '
        acc = np.array([])
        f_self = open('self_op.csv', 'w')
        f_self.write('c_op,-c,acc\n')
        for i in c:
            para = option + '-c ' + str(2 ** i)
            model = svmutil.svm_train(prob, para)
            acc = np.append(acc, model)
            f_self.write(str(i) + ',' + str(2 ** i) + ',' + str(model) + '\n')
        print(acc)
        print(c[np.argmax(acc)])
        best_para = ' -c ' + str(2 ** c[np.argmax(acc)])

        return best_para




# print(x_train_origin)
# print(x_train_origin.shape)
# print(rbf_kernel(x_train_origin, x_train_origin, 1/28))
# print(rbf_kernel(x_train_origin, x_train_origin, 1/28).shape)

# a = line_kernel(x_train_origin)
# print(a)
# print(a.shape)
"""
y_train = list(y_train)
y_test = list(y_test)

x_train_kernel = line_kernel(x_train_origin) + rbf_kernel(x_train_origin, 1/28)
x_test_kernel = line_kernel(x_test_origin) + rbf_kernel(x_test_origin, 1/28)
# print(x_train_kernel)
# print(x_train_kernel.shape)
x_train_self = convert_format(x_train_kernel)
x_test_self = convert_format(x_test_kernel)

prob_kernel = svmutil.svm_problem(y_train, x_train_self, isKernel=True)
# best_para_self = grid_search(prob_kernel, 0, [-9, -7, -5, -1, 1, 5, 8, 10, 15])
best_para_self = grid_search(prob_kernel, 4, [-7, 1, 5])
# best_para_self = ' -c 0.03125'
print(best_para_self)


# model = svmutil.svm_train(prob_kernel, best_para_self)
# p_label, p_acc, p_val = svmutil.svm_predict(y_test, x_test_self, model)
"""



# save_pkl_data(convert_format(x_train))
# x_train = load_pkl_data()

x_train = convert_format(x_train_origin)
x_test = convert_format(x_test_origin)
# y_train = list(y_train)
# y_test = list(y_test)



# grid_search(0, [-5,-3,-1,1,3,5,7])
prob = svmutil.svm_problem(y_train, x_train)


start = timeit.default_timer()

# # best_para_linear = grid_search(prob, 0, [-9, -7, -5, -1, 1, 5, 8, 10, 15])
# # best_para_linear = ' -c 0.03125'

# best_para_rbf = grid_search(prob, 2, [-7, -5, 1, 5, 10, 15], [1/1024, 1/784, 1/28, 1, 3])
# best_para_rbf = grid_search(prob, 2, [-7, 1, 10], [1/784, 1/28, 1, 10])
# # best_para_rbf = ' -c 32768 -g 0.03571428571428571'

# best_para = grid_search(prob, 0, [-5,-1,7])

best_para_poly = grid_search(prob, 1, [-7, 1, 10], [1/784, 1/28, 1], [-10, 1, 10], [2, 3, 4])

# best_para_poly = grid_search(prob, 1, [-7, 1, 10], [1/784, 1, 10], [1, 10], [2, 3, 4])


# model = svmutil.svm_train(prob, '-q -t 0 '+best_para_linear)
# p_label, p_acc, p_val = svmutil.svm_predict(y_test, x_test, model)
#
# model = svmutil.svm_train(prob, '-q -t 2 '+best_para_rbf)
# p_label, p_acc, p_val = svmutil.svm_predict(y_test, x_test, model)


stop = timeit.default_timer()
# print(best_para_linear)
# print(best_para_poly)
print('Time: ', stop - start)
## prob = svmutil.svm_problem(y_train, x_train)


#		self.c_begin, self.c_end, self.c_step = -5,  15,  2
#		self.g_begin, self.g_end, self.g_step =  3, -15, -2


# para_linear = svmutil.svm_parameter('-t 0 -b 1')
##para = '-t 0 -v 10 -c 1 -b 1'
##para_linear = svmutil.svm_parameter('-t 0 -v 10 -c 1')
# para_linear = svmutil.svm_parameter('-t 0 -c 1')
# print(para_linear)

##model = svmutil.svm_train(prob, para_linear)

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
