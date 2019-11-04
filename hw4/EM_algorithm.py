import numpy as np
import math
import pickle as pkl


def read_image(filename):
    input_file = open(filename, 'rb')
    magic_number = input_file.read(4)
    image_num = input_file.read(4)
    image_num = int.from_bytes(image_num, byteorder='big')
    rows = input_file.read(4)
    rows = int.from_bytes(rows, byteorder='big')
    columns = input_file.read(4)
    columns = int.from_bytes(columns, byteorder='big')

    img = np.zeros((image_num, rows, columns))

    # for num in range(100):
    for num in range(image_num):
        for row in range(rows):
            for column in range(columns):
                pixel = input_file.read(1)
                pixel = int.from_bytes(pixel, byteorder='big', signed=False)
                img[num][row][column] = int(pixel // 128)

    return img


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


def get_filename():
    filename_train_image = '/Users/yen/Downloads/train-images.idx3-ubyte'
    filename_train_label = '/Users/yen/Downloads/train-labels.idx1-ubyte'
    return filename_train_image, filename_train_label


def load_data():
    filename_train_image, filename_train_label = get_filename()
    train_image = read_image(filename_train_image)
    train_label = read_label(filename_train_label)

    return train_image, train_label


def save_pkl_data():
    filename_train_image, filename_train_label = get_filename()
    img = read_image(filename_train_image)
    with open('train_image.pkl', 'wb') as f:
        pkl.dump(img, f)
    with open('train_label.pkl', 'wb') as f:
        pkl.dump(read_label(filename_train_label), f)


def load_pkl_data():
    with open('train_image.pkl', 'rb') as f:
        train_image = pkl.load(f)
    with open('train_label.pkl', 'rb') as f:
        train_label = pkl.load(f)

    return train_image, train_label


train_image, train_label = load_pkl_data()


