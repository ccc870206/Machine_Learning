import numpy as np
import matplotlib.pyplot as plt
import math
import pickle as pkl
import scipy.stats


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


# def rescale_image_array(image_array):
#     for num in range(image_array.shape[0]):
#         for row in range(image_array.shape[1]):
#             for col in range(image_array.shape[2]):

"""!!! SHOULD DELETE!!!"""


def save_data(filename_train_image, filename_train_label, filename_test_image, filename_test_label):
    conti, discrete = read_image(filename_train_image)
    np.save('train_image_conti', conti)
    np.save('train_image_discrete', discrete)
    # print(discrete[0])
    print('train_image done')

    conti, discrete = read_image(filename_test_image)
    np.save('test_image_conti', conti)
    np.save('test_image_discrete', discrete)
    print('test_image done')

    np.save('train_label', read_label(filename_train_label))
    print('train_label done')

    np.save('test_label', read_label(filename_test_label))
    print('test_label done')


def load_np_data():
    train_image_conti = np.load('train_image_conti.npy')
    # print('train_image done', train_image_conti.shape)
    train_image_discrete = np.load('train_image_discrete.npy')
    # print('train_image done', train_image_discrete.shape)
    test_image_conti = np.load('test_image_conti.npy')
    # print('test_image done', test_image_conti.shape)
    test_image_discrete = np.load('test_image_discrete.npy')
    # print('test_image done', test_image_discrete.shape)
    train_label = np.load('train_label.npy')
    # print('train_labels done', train_label)
    test_label = np.load('test_label.npy')
    # print('test_labels done', test_label)

    # with open('train_image_conti.pkl', 'wb') as f:
    #     pkl.dump(train_image_conti, f)
    # with open('train_image_discrete.pkl', 'wb') as f:
    #     pkl.dump(train_image_discrete, f)
    # with open('test_image_conti.pkl', 'wb') as f:
    #     pkl.dump(test_image_conti, f)
    # with open('test_image_discrete.pkl', 'wb') as f:
    #     pkl.dump(test_image_discrete, f)
    # with open('train_label.pkl', 'wb') as f:
    #     pkl.dump(train_label, f)
    # with open('test_label.pkl', 'wb') as f:
    #     pkl.dump(test_label, f)

    return train_image_conti, train_image_discrete, test_image_conti, test_image_discrete, train_label, test_label


def save_pkl_data(filename_train_image, filename_train_label, filename_test_image, filename_test_label):
    conti, discrete = read_image(filename_train_image)
    with open('train_image_conti.pkl', 'wb') as f:
        pkl.dump(conti, f)
    with open('train_image_discrete.pkl', 'wb') as f:
        pkl.dump(discrete, f)
    conti, discrete = read_image(filename_test_image)
    with open('test_image_conti.pkl', 'wb') as f:
        pkl.dump(conti, f)
    with open('test_image_discrete.pkl', 'wb') as f:
        pkl.dump(discrete, f)
    with open('train_label.pkl', 'wb') as f:
        pkl.dump(read_label(filename_train_label), f)
    with open('test_label.pkl', 'wb') as f:
        pkl.dump(read_label(filename_test_label), f)


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
    # print(train_image)
    # print(dict_pixel_cond)
    for num in range(1, train_image.shape[0]):
        for row in range(train_image.shape[1]):
            for col in range(train_image.shape[2]):
                value = int(train_image[num][row][col])
                dict_pixel[row * 28 + col][value] += 1
                dict_pixel_cond[train_label[num][0]][row * 28 + col][value] += 1
                # print(train_label[num][0], row*28+col, value)
                # print()
                # print("pixel:",dict_pixel[row*28+col][value])
                # print("cond:",dict_pixel_cond[train_label[num][0]][row*28+col][value])
        dict_label[train_label[num][0]] += 1
    # print(dict_pixel_cond[0])
    return dict_pixel, dict_label, dict_pixel_cond


def compute_continuous_probability(train_image, train_label):
    all = (np.mean(train_image), np.std(train_image))
    total = 60000
    # train_label[0] = 0
    # train_label[1] = 0

    dict_label = {}
    dict_pixel = {}
    dict_pixel_cond = {}

    label_num = [0]*10
    label_sum = [0]*10
    label_square = [0]*10
    pixel_sum = [0]*28*28
    pixel_square = [0]*28*28
    pixel_cond_sum = [[0]*28*28]*10
    pixel_cond_sum = np.zeros((10, 28*28))
    pixel_cond_square = np.zeros((10, 28*28))


    label_arr = [[]] * 10

    test_0 = 0
    test_1 = 0
    # for num in range(train_image.shape[0]):
    # print(train_image[0])
    for num in range(total):
        # label_arr[int(train_label[num])].append(train_image[num])
        # print(label_arr[1])
        label_num[int(train_label[num])] += 1
        for row in range(train_image.shape[1]):
            for col in range(train_image.shape[2]):
                value = train_image[num][row][col]
                pixel_sum[row*28+col] += (value/total)
                pixel_square[row*28+col] += (value*value/total)
                pixel_cond_sum[int(train_label[num])][row*28+col] += value

                # print("v", value)
                # print(value*value)
                pixel_cond_square[int(train_label[num])][row * 28 + col] += (value*value)
                label_sum[int(train_label[num])] += value
                label_square[int(train_label[num])] += (value*value)
                # pixel_arr[row * 28 + col].append(train_image[num][row][col])
    # print("pixel cond:",pixel_cond_sum[0])
    # print("pixel cond1:", pixel_cond_sum[1])
    # print("pixel cond square:",pixel_cond_square[0])
                # pixel_cond_arr[int(train_label[num])][row * 28 + col].append(train_image[num][row][col])
    # print(pow(label_sum[0]/28/28/label_num[0], 2))
    # print(label_square[0]/28/28/label_num[0])
    # print(test_0, pow(sum(pixel_cond_sum[0])/28/28,2))
    # print(test_1, sum(pixel_cond_square[0]))
    # print(sum(pixel_sum)*sum(pixel_sum), sum(pixel_square))
    # print(pow(sum(pixel_square)-sum(pixel_sum)*sum(pixel_sum), 1/2))
    # print()
    test = 0
    # for i in pixel_cond_square[0]:
    #     test += i
    # print(test)
    # print(sum(pixel_cond_sum[0]), sum(pixel_cond_sum[0])*sum(pixel_cond_sum[0]))
    # print(sum(pixel_cond_square[0]))
    # print(pixel_cond_square[0])

    # print(sum(pixel_cond_sum[0]))
    # print(sum(pixel_cond_square[0]))

    for i in range(10):
        dict_label[i] = (label_sum[i] / 28 / 28 / label_num[i],
                         pow(label_square[i] / 28 / 28 / label_num[i] - pow(label_sum[i] / 28 / 28 / label_num[i], 2), 1/2))
        dict_pixel_cond[i] = {}
    # print(dict_pixel_cond)
    for i in range(10):
        for j in range(28*28):
            # print(pixel_cond_sum[i][j], label_num[i], pixel_cond_sum[i][j]/label_num[i])
            # print(label_square[i] / label_num[i])
            # print(label_square[i] / label_num[i] - pow(pixel_cond_sum[i][j]/label_num[i], 2))
            dict_pixel_cond[i][j] = (pixel_cond_sum[i][j]/label_num[i],
                                     pow(pixel_cond_square[i][j] / label_num[i] - pow(
                                         pixel_cond_sum[i][j]/label_num[i], 2), 1 / 2))
    # print(dict_pixel_cond)
        # print(label_num)
    #     label_sum[i] = sum([j/label_num[i]/28/28 for j in pixel_cond_sum[i]])
    #     label_square[i] = sum([j / label_num[i]/28/28 for j in pixel_cond_square[i]])
    #     print(label_sum[i])
    #     print(label_square[i])
    #     dict_label[i] = (label_sum[i], label_square[i] - label_sum[i]*label_sum[i])
    # #     # print("square", label_square[i])
    # #     # print("sum", label_sum[i]*label_sum[i])
    # print(dict_label)

    for i in range(28*28):
        dict_pixel[i] = (pixel_sum[i], np.power(pixel_square[i] - pixel_sum[i]*pixel_sum[i], 1/2))
    # print(dict_pixel)
    # print(dict_pixel)
    #     # dict_pixel[i] = (pixel_sum[i], np.power((pixel_square[i] - pixel_sum[i]*pixel_sum[i]), 1/2))
    # print(dict_pixel)
    # print(dict_pixel)
        # dict_pixel_cond[i] = {}
        # for j in range(28*28):
        #     dict_pixel_cond[i][j] = (np.mean(pixel_cond_arr[i][j]), np.std(pixel_cond_arr[i][j]))
    # for i in range(28*28):
    #     dict_pixel[i] = (np.mean(pixel_arr[i]), np.std(pixel_arr[i]))
    # print(len(label_arr),len(label_arr[]), len(label_arr[0][0]))
    return dict_pixel, dict_label, dict_pixel_cond, all


def gaussian_distribution(mean, std, x):
    # print(std)
    if std == 0:
        return 10e-7
    constant = 1/(std * pow(2*math.pi, 1/2))
    # if (std * pow(2*math.pi, 1/2)) == 0:
    #     print(std * pow(2*math.pi, 1/2))
    #     print(std)
    # print(constant)
    power = -(1/2) * pow((x-mean) / std, 2)
    # print(power)
    # print(constant * pow(math.e, power))
    return constant * pow(math.e, power)


def label_prediction_continuous(img_array, labels, dict_pixel, dict_label, dict_pixel_cond, all):
    total = 60000
    # total_cond = 0
    # total_pixel = 60000
    # total_pixel = 28*28
    # print(dict_label[int(label)])

    # p_pixel = np.zeros((28, 28))

    # p_pixel = 1
    # p_cond = np.zeros((28, 28))

    # p_cond = 1
    # p_posterior = np.zeros((28, 28))

    wrong = 0
    for i in range(10):
    # for i in range(10):
        img = img_array[i]
        ground_truth = labels[i]
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
                                              img[row][col])+10e-7))


            p_cond += math.log(p_label)
            # p_cond += math.log(1/10)
            # p_cond = p_cond * p_label


            posterior[label] = p_cond
        print(posterior)
            # print("p_pixel:", p_pixel)
            # print(label, p_cond)
        # print(posterior)
        posterior /= (np.sum(posterior))
        # print(posterior)
        # print(posterior)
        prediction = np.argmin(posterior)
        if prediction != ground_truth:
            wrong += 1
            print("prediction:", prediction, "ground_truth:", ground_truth)
        # print(np.argmin(posterior))
    print("wrong:", wrong)
    print("error rate:", wrong / 10000)
    # print(p_cond, math.log(p_cond))
    # print(p_pixel, math.log(p_pixel))
    # print(p_cond / p_pixel, math.log(p_cond) / math.log(p_pixel), math.log(p_cond / p_pixel))
    # print(pow(math.e, (p_cond / p_pixel)))

    # print(np.sum(p_posterior))
    # print("p_cond:", p_cond)
    # print("p_pixel:", p_pixel)


def pixel_prediction_continuous(dict_pixel, dict_label, dict_pixel_cond, all):
    p_pixel = 1
    label = 9

    imagination = np.zeros((28, 28))
    test_imagination = np.zeros((28, 28))
    # for row in range(4, 24):
    # p_label = gaussian_distribution(all[0], pow(all[1], 1/2), dict_label[label][0])
    # for row in range(1):
    for row in range(28):
        # row = 5
        # for col in range(4, 24):
        # for col in range(1):
        for col in range(28):
            # col = 4
            # posterior = np.empty(32)
            # p_pixel = gaussian_distribution(dict_pixel[label][0], dict_pixel[label][1], dict_label[label])



            # for pixel in range(32):
            #     # p_pixel = (math.log((dict_pixel_cond[label][row*28+col][pixel]+10e-7) /
            #     #                     (dict_pixel[row*28+col][pixel] + 10e-7)))
            #     p_pixel = gaussian_distribution(dict_pixel[label][0], dict_pixel[label][1], dict_label[label])
            #     # p_pixel = (math.log((dict_pixel_cond[label][row * 28 + col][pixel] + 1) / dict_label[label]))
            #     # print("cond:", (dict_pixel_cond[label][row * 28 + col][pixel] + 1))
            #     # print("all:", (dict_pixel[row*28+col][pixel] + 10e-7))
            #     # print("all:", p_label)
            #     # print("log:", p_pixel)
            #     # p_pixel = (math.log((dict_pixel_cond[label][row * 28 + col][pixel] + 10e-7) / (1/32)))
            #     # print((dict_pixel_cond[label][row*28+col][pixel]+10e-7) / (dict_pixel[row*28+col][pixel] + 10e-7))
            #     # p_pixel += math.log(dict_pixel[row*28+col][pixel] + 10e-7)
            #     # p_pixel += math.log(dict_pixel[row*28+col][pixel] + 10e-7)
            #     p_pixel += math.log(p_label)
            #     # print("log 1/32", math.log(1/32))
            #     print("log p_label", math.log(1 / 32))
            #     posterior[pixel] = p_pixel


            #
            # posterior /= (np.sum(posterior))
            # # print(row, col)
            # # print(np.argmin(posterior))
            # print(posterior)
            # test_imagination[row][col] = np.argmin(posterior)
            if dict_pixel_cond[label][row*28+col][0] > 127:
                imagination[row][col] = 1
    # print(test_imagination)
    plt.imshow(imagination, 'Greys')
    plt.show()


def label_prediction_discrete(img_array, labels, dict_pixel, dict_label, dict_pixel_cond):
    total = 60000
    # total_cond = 0
    # total_pixel = 60000
    # total_pixel = 28*28
    # print(dict_label[int(label)])

    # p_pixel = np.zeros((28, 28))

    # p_pixel = 1
    # p_cond = np.zeros((28, 28))

    # p_cond = 1
    # p_posterior = np.zeros((28, 28))

    wrong = 0
    # for i in range(10000):
    for i in range(10000):
        img = img_array[i]
        ground_truth = labels[i]
        posterior = np.empty(10)
        for label in range(10):
            p_pixel = 0
            p_cond = 0
            p_label = dict_label[int(label)] / total
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
                    p_pixel += (math.log((dict_pixel[row * 28 + col][img[row][col]] + 10e-7) / total))
                    # print(dict_pixel_cond[int(label)][row*28+col][img[row][col]] / dict_label[int(label)])
                    p_cond += (math.log(
                        (dict_pixel_cond[int(label)][row * 28 + col][img[row][col]] + 10e-7) / dict_label[int(label)]))
            p_cond += math.log(p_label)
            # p_cond += math.log(1/10)
            # p_cond = p_cond * p_label
            posterior[label] = p_cond
            # print("p_pixel:", p_pixel)
            # print(label, p_cond)

        posterior /= (np.sum(posterior))
        # print(posterior)
        prediction = np.argmin(posterior)
        if prediction != ground_truth:
            wrong += 1
            print("prediction:", prediction, "ground_truth:", ground_truth)
        # print(np.argmin(posterior))
    print("wrong:", wrong)
    print("error rate:", wrong / 10000)
    # print(p_cond, math.log(p_cond))
    # print(p_pixel, math.log(p_pixel))
    # print(p_cond / p_pixel, math.log(p_cond) / math.log(p_pixel), math.log(p_cond / p_pixel))
    # print(pow(math.e, (p_cond / p_pixel)))

    # print(np.sum(p_posterior))
    # print("p_cond:", p_cond)
    # print("p_pixel:", p_pixel)


def pixel_prediction_discrete(dict_pixel, dict_label, dict_pixel_cond):
    p_pixel = 1
    label = 4

    imagination = np.zeros((28, 28))
    test_imagination = np.zeros((28, 28))
    # for row in range(4, 24):
    p_label = dict_label[label] / 60000
    # for row in range(1):
    for row in range(28):
        # row = 5
        # for col in range(4, 24):
        # for col in range(1):
        for col in range(28):
            # col = 4
            posterior = np.empty(32)
            for pixel in range(32):
                # p_pixel = (math.log((dict_pixel_cond[label][row*28+col][pixel]+10e-7) /
                #                     (dict_pixel[row*28+col][pixel] + 10e-7)))
                p_pixel = (math.log((dict_pixel_cond[label][row * 28 + col][pixel] + 1) / dict_label[label]))
                print("cond:", (dict_pixel_cond[label][row * 28 + col][pixel] + 1))
                # print("all:", (dict_pixel[row*28+col][pixel] + 10e-7))
                print("all:", p_label)
                print("log:", p_pixel)
                # p_pixel = (math.log((dict_pixel_cond[label][row * 28 + col][pixel] + 10e-7) / (1/32)))
                # print((dict_pixel_cond[label][row*28+col][pixel]+10e-7) / (dict_pixel[row*28+col][pixel] + 10e-7))
                # p_pixel += math.log(dict_pixel[row*28+col][pixel] + 10e-7)
                # p_pixel += math.log(dict_pixel[row*28+col][pixel] + 10e-7)
                p_pixel += math.log(p_label)
                # print("log 1/32", math.log(1/32))
                print("log p_label", math.log(1 / 32))
                posterior[pixel] = p_pixel
            posterior /= (np.sum(posterior))
            print(row, col)
            print(np.argmin(posterior))
            print(posterior)
            test_imagination[row][col] = np.argmin(posterior)
            if np.argmin(posterior) > 15:
                imagination[row][col] = 1
    # print(test_imagination)
    plt.imshow(imagination, 'Greys')
    plt.show()


filename_train_image = '/Users/yen/Downloads/train-images.idx3-ubyte'
filename_train_label = '/Users/yen/Downloads/train-labels.idx1-ubyte'
filename_test_image = '/Users/yen/Downloads/t10k-images.idx3-ubyte'
filename_test_label = '/Users/yen/Downloads/t10k-labels.idx1-ubyte'

# save_data(filename_train_image, filename_train_label, filename_test_image, filename_test_label)

# train_image_conti, train_image_discrete, test_image_conti, test_image_discrete, train_label, test_label = load_np_data()
train_image_conti, train_image_discrete, test_image_conti, test_image_discrete, train_label, test_label = load_pkl_data()

# dict_pixel, dict_label, dict_pixel_cond, all = compute_continuous_probability(train_image_conti, train_label)
# with open('dict_pixel_conti.pkl', 'wb') as f:
#     pkl.dump(dict_pixel, f)
# with open('dict_label_conti.pkl', 'wb') as f:
#     pkl.dump(dict_label, f)
# with open('dict_pixel_cond_conti.pkl', 'wb') as f:
#     pkl.dump(dict_pixel_cond, f)
# with open('all_conti.pkl', 'wb') as f:
#     pkl.dump(all, f)
# print(np.mean(train_image_conti[0]), np.std(train_image_conti[0]))
with open('dict_pixel_conti.pkl', 'rb') as f:
    dict_pixel = pkl.load(f)
with open('dict_label.pkl', 'rb') as f:
    dict_label = pkl.load(f)
with open('dict_pixel_cond_conti.pkl', 'rb') as f:
    dict_pixel_cond = pkl.load(f)
with open('all_conti.pkl', 'rb') as f:
    all = pkl.load(f)

dict_label = dict_label.item()
# print(dict_pixel_cond)
# print(dict_pixel_cond[0])
# print(dict_pixel_cond[0])
# print(dict_pixel)
# print(dict_label)

# print(train_image_conti[0])
# print(dict_pixel_cond)
# label_prediction_continuous(test_image_conti, test_label, dict_pixel, dict_label, dict_pixel_cond, all)
pixel_prediction_continuous(dict_pixel, dict_label, dict_pixel_cond, all)
# print(gaussian_distribution(0, 1, 0))

# y = scipy.stats.norm.pdf(0,0,1)
# print(y)
# dict_pixel, dict_label, dict_pixel_cond = compute_continuous_probability(train_image_conti, train_label)

#
#
# dict_pixel, dict_label, dict_pixel_cond = compute_discrete_probability(train_image_discrete, train_label)
#
# np.save('dict_pixel', dict_pixel)
# np.save('dict_label', dict_label)
# np.save('dict_pixel_cond', dict_pixel_cond)

# # DISCRETE DATA
# dict_pixel = np.load('dict_pixel.npy', allow_pickle=True).item()
# dict_label = np.load('dict_label.npy', allow_pickle=True).item()
# dict_pixel_cond = np.load('dict_pixel_cond.npy', allow_pickle=True).item()

# dict_pixel = np.load('dict_pixel.npy', allow_pickle=True)
# dict_label = np.load('dict_label.npy', allow_pickle=True)
# dict_pixel_cond = np.load('dict_pixel_cond.npy', allow_pickle=True)
# with open('dict_pixel.pkl', 'wb') as f:
#     pkl.dump(dict_pixel, f)
# with open('dict_label.pkl', 'wb') as f:
#     pkl.dump(dict_label, f)
# with open('dict_pixel_cond.pkl', 'wb') as f:
#     pkl.dump(dict_pixel_cond, f)

# with open('dict_pixel.pkl', 'rb') as f:
#     dict_pixel = pkl.load(f).item()
# with open('dict_label.pkl', 'rb') as f:
#     dict_label = pkl.load(f).item()
# with open('dict_pixel_cond.pkl', 'rb') as f:
#     dict_pixel_cond = pkl.load(f).item()

# print(dict_pixel_cond)


# COUNT EMPTY
# count = 0
# zero_count = 0
# for key1, value1 in dict_pixel_cond.items():
#     for key2, value2 in value1.items():
#         for key3, value3 in value2.items():
#             count += 1
#             print(value2[0])
#             if value2[0] != dict_label[key1] and value3 == 0:
#                 zero_count += 1
# 38635
# 103332
# 250880


# for key2, value2 in dict_pixel.items():
#     for key3, value3 in value2.items():
#         count += 1
#         if value2[0] != 60000 and value3 == 0:
#             zero_count += 1
# print(dict_pixel)
# print(count)
# print(zero_count)
# 2881
# 4989
# 25098


# print(dict_pixel[7])
# print(dict_pixel_cond[7])
# label_prediction_discrete(test_image_discrete, test_label, dict_pixel, dict_label, dict_pixel_cond)
# pixel_prediction_discrete(dict_pixel, dict_label, dict_pixel_cond)

# SPEC PIC
# str_9 = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
# str_0 = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
#
# arr = str.split(" ")
# arr_np = np.array(arr, dtype=int).reshape((28,28))
# print(arr_np.shape)
# arr_np = np.reshape(arr_np, (28, 28))
#
# print(arr_np)
# plt.imshow(arr_np, 'Greys')
# plt.show()


# print(dict_pixel)
# print(dict_label)
# print(dict_pixel_cond[0])

# for row in range(test_image_discrete.shape[0]):
#     for col in range(test_image_discrete[1]):


# plt.imshow(test_image_discrete[0], 'Greys')
# plt.show()
