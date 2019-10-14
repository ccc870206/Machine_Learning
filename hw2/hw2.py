import numpy as np
import math
import pickle as pkl
import timeit


def read_image(filename):
    input_file = open(filename, 'rb')
    magic_number = input_file.read(4)
    image_num = input_file.read(4)
    image_num = int.from_bytes(image_num, byteorder='big')
    rows = input_file.read(4)
    rows = int.from_bytes(rows, byteorder='big')
    columns = input_file.read(4)
    columns = int.from_bytes(columns, byteorder='big')

    img_conti = np.zeros((image_num, rows, columns))
    img_discrete = np.zeros((image_num, rows, columns))

    # for num in range(1):
    for num in range(image_num):
        for row in range(rows):
            for column in range(columns):
                pixel = input_file.read(1)
                pixel = int.from_bytes(pixel, byteorder='big', signed=False)
                img_conti[num][row][column] = pixel
                img_discrete[num][row][column] = int(pixel // 8)

    return img_conti, img_discrete


def read_label(filename):
    input_file = open(filename, 'rb')
    magic_number = input_file.read(4)
    image_num = input_file.read(4)
    image_num = int.from_bytes(image_num, byteorder='big')

    labels = np.zeros((image_num, 1))

    for num in range(image_num):
        label = input_file.read(1)
        label = int.from_bytes(label, byteorder='big', signed=False)
        labels[num] = label
    return labels


def load_data(train_image, train_label, test_image, test_label):
    train_image_conti, train_image_discrete = read_image(train_image)
    test_image_conti, test_image_discrete = read_image(test_image)
    train_label = read_label(train_label)
    test_label = read_label(test_label)

    return train_image_conti, train_image_discrete, test_image_conti, test_image_discrete, train_label, test_label


def save_pkl_data(train_image, train_label, test_image, test_label):
    conti, discrete = read_image(train_image)
    with open('train_image_conti.pkl', 'wb') as f:
        pkl.dump(conti, f)
    with open('train_image_discrete.pkl', 'wb') as f:
        pkl.dump(discrete, f)
    conti, discrete = read_image(test_image)
    with open('test_image_conti.pkl', 'wb') as f:
        pkl.dump(conti, f)
    with open('test_image_discrete.pkl', 'wb') as f:
        pkl.dump(discrete, f)
    with open('train_label.pkl', 'wb') as f:
        pkl.dump(read_label(train_label), f)
    with open('test_label.pkl', 'wb') as f:
        pkl.dump(read_label(test_label), f)


def load_pkl_data():
    with open('train_image_conti.pkl', 'rb') as f:
        train_image_conti = pkl.load(f)
    with open('train_image_discrete.pkl', 'rb') as f:
        train_image_discrete = pkl.load(f)
    with open('test_image_conti.pkl', 'rb') as f:
        test_image_conti = pkl.load(f)
    with open('test_image_discrete.pkl', 'rb') as f:
        test_image_discrete = pkl.load(f)
    with open('train_label.pkl', 'rb') as f:
        train_label = pkl.load(f)
    with open('test_label.pkl', 'rb') as f:
        test_label = pkl.load(f)
    return train_image_conti, train_image_discrete, test_image_conti, test_image_discrete, train_label, test_label


def load_discrete_dict():
    with open('dict_pixel.pkl', 'rb') as f:
        dict_pixel = pkl.load(f).item()
    with open('dict_label.pkl', 'rb') as f:
        dict_label = pkl.load(f).item()
    with open('dict_pixel_cond.pkl', 'rb') as f:
        dict_pixel_cond = pkl.load(f).item()
    return dict_pixel, dict_label, dict_pixel_cond


def load_continuous_dict():
    with open('dict_pixel_conti.pkl', 'rb') as f:
        dict_pixel = pkl.load(f)
    with open('dict_label.pkl', 'rb') as f:
        dict_label = pkl.load(f)
    with open('dict_pixel_cond_conti.pkl', 'rb') as f:
        dict_pixel_cond = pkl.load(f)
    with open('all_conti.pkl', 'rb') as f:
        all = pkl.load(f)
    dict_label = dict_label.item()
    return dict_pixel, dict_label, dict_pixel_cond, all


def compute_discrete_probability(train_image, train_label):
    dict_pixel = {}
    dict_label = {}
    dict_pixel_cond = {}

    for i in range(10):
        dict_label[i] = 0
        dict_pixel_cond[i] = {}
    for row in range(train_image.shape[1]):
        for col in range(train_image.shape[2]):
            value = int(train_image[0][row][col])
            dict_pixel[row * 28 + col] = {value: 1}
            dict_pixel_cond[train_label[0][0]][row * 28 + col] = {value: 1}

            for i in range(32):
                dict_pixel[row * 28 + col][i] = dict_pixel[row * 28 + col].get(i, 0)
                for j in range(10):
                    dict_pixel_cond[j][row * 28 + col] = dict_pixel_cond[j].get(row * 28 + col, {})
                    dict_pixel_cond[j][row * 28 + col][i] = \
                        dict_pixel_cond[j][row * 28 + col].get(i, 0)

    dict_label[train_label[0][0]] += 1

    for num in range(1, train_image.shape[0]):
        for row in range(train_image.shape[1]):
            for col in range(train_image.shape[2]):
                value = int(train_image[num][row][col])
                dict_pixel[row * 28 + col][value] += 1
                dict_pixel_cond[train_label[num][0]][row * 28 + col][value] += 1

        dict_label[train_label[num][0]] += 1

    return dict_pixel, dict_label, dict_pixel_cond


def compute_continuous_probability(train_image, train_label):
    all = (np.mean(train_image), np.std(train_image))
    total = 60000

    dict_label = {}
    dict_pixel = {}
    dict_pixel_cond = {}

    label_num = [0]*10
    label_sum = [0]*10
    label_square = [0]*10
    pixel_sum = [0]*28*28
    pixel_square = [0]*28*28
    pixel_cond_sum = np.zeros((10, 28*28))
    pixel_cond_square = np.zeros((10, 28*28))

    for num in range(total):
        label_num[int(train_label[num])] += 1
        for row in range(train_image.shape[1]):
            for col in range(train_image.shape[2]):
                value = train_image[num][row][col]
                pixel_sum[row*28+col] += (value/total)
                pixel_square[row*28+col] += (value*value/total)
                pixel_cond_sum[int(train_label[num])][row*28+col] += value

                pixel_cond_square[int(train_label[num])][row * 28 + col] += (value*value)
                label_sum[int(train_label[num])] += value
                label_square[int(train_label[num])] += (value*value)

    # construct dict_label
    for i in range(10):
        dict_label[i] = (label_sum[i] / 28 / 28 / label_num[i],
                         pow(label_square[i] / 28 / 28 / label_num[i] - pow(label_sum[i] / 28 / 28 / label_num[i], 2), 1/2))
        dict_pixel_cond[i] = {}
    # construct dict_pixel_cond
    for i in range(10):
        for j in range(28*28):
            dict_pixel_cond[i][j] = (pixel_cond_sum[i][j]/label_num[i],
                                     pow(pixel_cond_square[i][j] / label_num[i] - pow(
                                         pixel_cond_sum[i][j]/label_num[i], 2), 1 / 2))
    # construct dict_pixel
    for i in range(28*28):
        dict_pixel[i] = (pixel_sum[i], np.power(pixel_square[i] - pixel_sum[i]*pixel_sum[i], 1/2))

    return dict_pixel, dict_label, dict_pixel_cond, all


def gaussian_distribution(mean, std, x):

    if std == 0:
        return 10e-7
    constant = 1/(std * pow(2*math.pi, 1/2))
    power = -(1/2) * pow((x-mean) / std, 2)

    return constant * pow(math.e, power)


def output_imagination(imagination):
    for i in range(28):
        print(" ".join(np.array(imagination[i], dtype=str)))


def output_posterior(posterior):
    print("Posterior (in log scale)")
    for i in range(10):
        print(str(i)+":", posterior[i])


def label_prediction_discrete(img_array, labels, dict_pixel, dict_label, dict_pixel_cond):
    total = 60000
    wrong = 0

    for i in range(10000):
        img = img_array[i]
        ground_truth = labels[i][0]
        posterior = np.empty(10)
        for label in range(10):
            p_pixel = 0
            p_cond = 0
            p_label = dict_label[int(label)] / total

            for row in range(img.shape[0]):
                for col in range(img.shape[1]):
                    p_pixel += (math.log((dict_pixel[row * 28 + col][img[row][col]] + 10e-7) / total))
                    p_cond += (math.log(
                        (dict_pixel_cond[int(label)][row * 28 + col][img[row][col]] + 10e-7) / dict_label[int(label)]))
            p_cond += math.log(p_label)
            posterior[label] = p_cond

        posterior /= (np.sum(posterior))
        output_posterior(posterior)
        prediction = np.argmin(posterior)
        if prediction != ground_truth:
            wrong += 1
        print("Prediction:", prediction, ", Ans:", int(ground_truth))
        print()
    return wrong / 10000


def pixel_prediction_discrete(dict_pixel, dict_label, dict_pixel_cond):

    for label in  range(10):
        p_label = dict_label[label] / 60000
        imagination = np.zeros((28, 28))
        for row in range(28):
            for col in range(28):
                posterior = np.empty(32)
                for pixel in range(32):
                    p_pixel = (math.log((dict_pixel_cond[label][row * 28 + col][pixel] + 1) / dict_label[label]))
                    p_pixel += math.log(p_label)

                    posterior[pixel] = p_pixel
                posterior /= (np.sum(posterior))
                if np.argmin(posterior) > 15:
                    imagination[row][col] = 1

        print(str(label), ":")
        output_imagination(np.array(imagination, dtype=int))
        print()


def label_prediction_continuous(img_array, labels, dict_pixel, dict_label, dict_pixel_cond, all):
    total = 60000

    wrong = 0
    for i in range(10):

        img = img_array[i]
        ground_truth = labels[i][0]
        posterior = np.empty(10)
        for label in range(10):
            p_pixel = 0
            p_cond = 0
            # p_label = dict_label[label] / total
            # print(dict_label[1])
            p_label = dict_label[int(label)] / total
            # p_label = gaussian_distribution(all[0], pow(all[1], 1/2), dict_label[label][0])
            # print(pow(all[1], 1/2))
            # print(dict_label[label][1])
            # print("p_label:", p_label)
            # print("p_label:", p_label)
            for row in range(img.shape[0]):
                for col in range(img.shape[1]):
                    """
                    # print(row, col)
                    # print('dict pixel cond:', dict_pixel_cond[int(label)][row*28+col][img[row][col]])
                    # print('label:', dict_label[int(label)])
                    p_cond[row][col] = dict_pixel_cond[int(label)][row*28+col][img[row][col]] / dict_label[int(label)]
                    p_pixel[row][col] = dict_pixel[row*28+col][img[row][col]] / total
                    # print(p_pixel[row][col])
                    p_posterior[row][col] = p_cond[row][col]*p_label / p_pixel[row][col]
                    """
                    value = img[row][col]
                    # p_pixel = p_pixel * dict_pixel[row*28+col][img[row][col]] / total
                    # p_cond = p_cond * dict_pixel_cond[int(label)][row*28+col][img[row][col]] / dict_label[int(label)]
                    # print("pixel", math.log(dict_pixel[row*28+col][value] / total))
                    # print("cond", math.log(dict_pixel_cond[int(label)][row*28+col][value] / dict_label[int(label)]))

                    # p_pixel += (math.log((dict_pixel[row * 28 + col][img[row][col]] + 10e-7) / total))

                    # print(dict_pixel_cond[int(label)][row*28+col][img[row][col]] / dict_label[int(label)])
                    # print(gaussian_distribution(dict_pixel_cond[int(label)][row * 28 + col][0],
                    #                           dict_pixel_cond[int(label)][row * 28 + col][1],
                    #                           img[row][col])+10e-7)
                    p_cond += (math.log(
                        gaussian_distribution(dict_pixel_cond[int(label)][row * 28 + col][0],
                                              dict_pixel_cond[int(label)][row * 28 + col][1],
                                              img[row][col]) + 10e-7))

            p_cond += math.log(p_label)
            # p_cond += math.log(1/10)
            # p_cond = p_cond * p_label

            posterior[label] = p_cond
        # print(posterior)
        # print("p_pixel:", p_pixel)
        # print(label, p_cond)
        # print(posterior)
        posterior /= (np.sum(posterior))
        output_posterior(posterior)
        # print(posterior)
        # print(posterior)
        prediction = np.argmin(posterior)
        if prediction != ground_truth:
            wrong += 1

        print("Prediction:", prediction, ", Ans:", int(ground_truth))
        print()
    return wrong / 10000
    # print(p_cond, math.log(p_cond))
    # print(p_pixel, math.log(p_pixel))
    # print(p_cond / p_pixel, math.log(p_cond) / math.log(p_pixel), math.log(p_cond / p_pixel))
    # print(pow(math.e, (p_cond / p_pixel)))

    # print(np.sum(p_posterior))
    # print("p_cond:", p_cond)
    # print("p_pixel:", p_pixel)


def pixel_prediction_continuous(dict_pixel, dict_label, dict_pixel_cond, all):
    for label in range(10):
        imagination = np.zeros((28, 28))
        for row in range(28):
            for col in range(28):
                if dict_pixel_cond[label][row*28+col][0] > 127:
                    imagination[row][col] = 1
        print(label, ":")
        output_imagination(np.array(imagination, dtype=int))
        print()


def discrete_mode():
    # Load preprocessing data
    _, train_image_discrete, _, test_image_discrete, train_label, test_label = load_pkl_data()
    dict_pixel, dict_label, dict_pixel_cond = load_discrete_dict()

    # Load raw data and process
    # _, train_image_discrete, _, test_image_discrete, train_label, test_label = load_data()
    # dict_pixel, dict_label, dict_pixel_cond = compute_discrete_probability(train_image_discrete, train_label)

    error_rate = label_prediction_discrete(test_image_discrete, test_label, dict_pixel, dict_label, dict_pixel_cond)
    print("Imagination of numbers in Bayesian classifier:")
    pixel_prediction_discrete(dict_pixel, dict_label, dict_pixel_cond)
    print("Error rate:", error_rate)


def continuous_mode():
    # Load preprocessing data
    train_image_conti, _, test_image_conti, _, train_label, test_label = load_pkl_data()
    dict_pixel, dict_label, dict_pixel_cond, all = load_continuous_dict()
    
    # Load raw data and process
    # train_image_conti, _, test_image_conti, _, train_label, test_label = load_data()
    # dict_pixel, dict_label, dict_pixel_cond, all = compute_continuous_probability(train_image_conti, train_label)
    error_rate = label_prediction_continuous(test_image_conti, test_label, dict_pixel, dict_label, dict_pixel_cond, all)
    print("Imagination of numbers in Bayesian classifier:")
    pixel_prediction_continuous(dict_pixel, dict_label, dict_pixel_cond, all)
    print("Error rate:", error_rate)


if __name__ == "__main__":
    filename_train_image = '/Users/yen/Downloads/train-images.idx3-ubyte'
    filename_train_label = '/Users/yen/Downloads/train-labels.idx1-ubyte'
    filename_test_image = '/Users/yen/Downloads/t10k-images.idx3-ubyte'
    filename_test_label = '/Users/yen/Downloads/t10k-labels.idx1-ubyte'
    option = int(input("option:"))
    start = timeit.default_timer()
    if option == 0:
        discrete_mode()
    else:
        continuous_mode()
    stop = timeit.default_timer()

    print('Time: ', stop - start)