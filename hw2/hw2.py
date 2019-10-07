import numpy as np
import matplotlib.pyplot as plt


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

    # plt.imshow(img_discrete[0], 'Greys')
    # plt.show()
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


def load_data():
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
    return train_image_conti, train_image_discrete, test_image_conti, test_image_discrete, train_label, test_label


def compute_probability(train_image, train_label):
    dict_pixel = {}
    dict_label = {}
    dict_pixel_cond ={}

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
                    dict_pixel_cond[j][row * 28 + col] = dict_pixel_cond[j].get(0, {})
                    dict_pixel_cond[j][row * 28 + col][i] = \
                        dict_pixel_cond[j][row * 28 + col].get(i, 0)

    dict_label[train_label[0][0]] += 1

    # print(train_image)
    # print(dict_pixel_cond)
    for num in range(1, train_image.shape[0]):
        for row in range(train_image.shape[1]):
            for col in range(train_image.shape[2]):
                value = int(train_image[num][row][col])
                dict_pixel[row*28+col][value] += 1
                dict_pixel_cond[train_label[num][0]][row*28+col][value] += 1
                # print(train_label[num][0], row*28+col, value)
                # print()
                print("pixel:",dict_pixel[row*28+col][value])
                print("cond:",dict_pixel_cond[train_label[num][0]][row*28+col][value])
        dict_label[train_label[num][0]] += 1
    # print(dict_pixel_cond[0])
    return dict_pixel, dict_label, dict_pixel_cond


filename_train_image = '/Users/yen/Downloads/train-images.idx3-ubyte'
filename_train_label = '/Users/yen/Downloads/train-labels.idx1-ubyte'
filename_test_image = '/Users/yen/Downloads/t10k-images.idx3-ubyte'
filename_test_label = '/Users/yen/Downloads/t10k-labels.idx1-ubyte'

# save_data(filename_train_image, filename_train_label, filename_test_image, filename_test_label)


load_data()
train_image_conti, train_image_discrete, test_image_conti, test_image_discrete, train_label, test_label = load_data()


dict_pixel, dict_label, dict_pixel_cond = compute_probability(train_image_discrete[:][:][:10], train_label)

# np.save('dict_pixel', dict_pixel)
# np.save('dict_label', dict_label)
# np.save('dict_pixel_cond', dict_pixel_cond)

# dict_pixel = np.load('dict_pixel.npy', allow_pickle=True).item()
# dict_label = np.load('dict_label.npy', allow_pickle=True).item()
# dict_pixel_cond = np.load('dict_pixel_cond.npy', allow_pickle=True).item()
# print(dict_pixel)
# print(dict_label)
# print(dict_pixel_cond)

# for row in range(test_image_discrete.shape[0]):
#     for col in range(test_image_discrete[1]):


# plt.imshow(test_image_discrete[0], 'Greys')
# plt.show()
