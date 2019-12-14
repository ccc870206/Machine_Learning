import libsvm.python.svmutil as svmutil
import numpy as np
from scipy.spatial.distance import pdist, squareform


def load_data():
    x_train_origin = np.genfromtxt('X_train.csv', delimiter=',')
    y_train = np.genfromtxt('Y_train.csv', delimiter=',')
    x_test_origin = np.genfromtxt('X_test.csv', delimiter=',')
    y_test = np.genfromtxt('Y_test.csv', delimiter=',')
    y_train = list(y_train)
    y_test = list(y_test)
    return x_train_origin, y_train, x_test_origin, y_test


def find_best_parameter(x_origin, y):
    x = convert_format(x_origin)
    prob = svmutil.svm_problem(y, x)
    linear = grid_search(prob, 0, [-9, -7, -5, -1, 1, 5, 8, 10, 15])
    poly = grid_search(prob, 1, [-7, 1, 10], [1 / 784, 1 / 28, 1], [-10, 1, 10], [2, 3, 4])
    rbf = grid_search(prob, 2, [-7, -5, 1, 5, 10, 15], [1 / 1024, 1 / 784, 1 / 28, 1, 3])
    self, gamma_self = grid_search_self(x_origin, [-7, -5, 1, 5, 10, 15], [1/1024, 1/784, 1/28, 1, 3])

    return linear, poly, rbf, self, gamma_self, prob


def convert_format(data):
    x = [{i+1:data[j][i] for i in range(data.shape[1])} for j in range(data.shape[0])]
    return x


def convert_kernel_format(data):
    kernel = np.hstack((np.arange(1, len(data) + 1).reshape(-1, 1), data))
    return kernel


def line_kernel(x):
    return np.matmul(x, x.T)


def poly_kernel(x, c=1, d=2):
    term = c + np.matmul(x)
    return np.power(term, d)


def rbf_kernel(x, gamma):
    return squareform(np.exp(-gamma * pdist(x, 'sqeuclidean')))


def grid_search(prob, type, c, g=[], r=[], d=[]):
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

        best_para = ' -c ' + str(2**c[idx[0]]) + ' -g ' + str(g[idx[1]])

        return best_para


def grid_search_self(data, c, g):
    option = '-q -t 4 -v 10 '
    acc = np.array([])
    f_self = open('self_op.csv', 'w')
    f_self.write('c_op,-c,acc\n')
    for j in g:
        x_train_kernel = line_kernel(data) + rbf_kernel(data, j)
        x_train_self = convert_kernel_format(x_train_kernel)
        prob_kernel = svmutil.svm_problem(y_train, x_train_self, isKernel=True)
        for i in c:
            para = option + '-c ' + str(2 ** i)
            model = svmutil.svm_train(prob_kernel, para)
            acc = np.append(acc, model)
            f_self.write(str(i) + ',' + str(2 ** i) + ',' + str(model) + '\n')

    acc = acc.reshape((len(c), len(g)))
    idx = np.unravel_index(np.argmax(acc), acc.shape)

    best_para = ' -c ' + str(2**c[idx[0]])
    best_g = g[idx[1]]
    return best_para, best_g


if __name__ == "__main__":
    x_train_origin, y_train, x_test_origin, y_test = load_data()
    x_test = convert_format(x_test_origin)

    # find the best parameters by grid search
    bp_linear, bp_poly, bp_rbf, bp_self, bp_gamma_self, prob = find_best_parameter(x_train_origin, y_train)

    # prediction (linear / poly / rbf)
    model = svmutil.svm_train(prob, '-q -t 0 '+bp_linear)
    p_label, p_acc, p_val = svmutil.svm_predict(y_test, x_test, model)
    model = svmutil.svm_train(prob, '-q -t 1 '+bp_poly)
    p_label, p_acc, p_val = svmutil.svm_predict(y_test, x_test, model)
    model = svmutil.svm_train(prob, '-q -t 2 '+bp_rbf)
    p_label, p_acc, p_val = svmutil.svm_predict(y_test, x_test, model)

    # prediction (linear + rbf)
    x_test_kernel = line_kernel(x_test_origin) + rbf_kernel(x_test_origin, bp_gamma_self)
    x_test_self = convert_kernel_format(x_test_kernel)
    x_train_kernel = line_kernel(x_train_origin) + rbf_kernel(x_train_origin, bp_gamma_self)
    x_train_self = convert_kernel_format(x_train_kernel)
    prob_kernel = svmutil.svm_problem(y_train, x_train_self, isKernel=True)
    model = svmutil.svm_train(prob_kernel, '-t 4 '+bp_self)
    p_label, p_acc, p_val = svmutil.svm_predict(y_test, x_test_self, model)
