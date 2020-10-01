import pickle
import random
import cv2
import numpy as np


def load_data(dir_path):
    """
    Function to load train,valid and test data
    :param dir_path: absolute directory path to train, valid and test data set
    :return Train,valid, test data
    """
    training_file = (dir_path + '/train.p')
    validation_file = (dir_path + '/valid.p')
    testing_file = (dir_path + '/test.p')
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    print("Data Loaded")

    return train, valid, test


def data_summary(train, valid, test):
    """
    Function to provide a basic summary of Data set using python numpy and pandas
    """
    x_train, y_train = train['features'], train['labels']
    x_valid, y_valid = valid['features'], valid['labels']
    x_test, y_test = test['features'], test['labels']
    n_train = len(x_train)
    n_test = len(x_test)
    n_valid = len(x_valid)
    image_shape = x_train[0].shape
    num_classes = len(np.unique(y_train))

    print('No.of Training Examples: ', n_train)
    print('No.of validation Examples: ', n_valid)
    print('No.of Test Examples: ', n_test)
    print('Image Data Shape: ', image_shape)
    print('No.of classes: ', num_classes)

    return n_train, n_valid, n_test, image_shape, num_classes


def aug_dataset(dataset, dataset_class, deficit, deficit_class):
    x_train_deficit = dataset[dataset_class == deficit_class]
    x_train_augment = []
    for i in range(deficit):
        index = random.randint(0, len(x_train_deficit) - 1)
        x_train_augment.append(x_train_deficit[index])
    return x_train_augment, deficit_class * np.ones(len(x_train_augment))


def rand_translation(dataset):
    rand_translation_dataset = []
    rows, cols = dataset[0].shape[: 2]
    for i in range(len(dataset)):
        img = dataset[i]
        delx, dely = np.random.randint(-3, 3, 2)
        M = np.float32([[1, 0, delx], [0, 1, dely]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        rand_translation_dataset.append(dst)
    return rand_translation_dataset


def rand_rotation(dataset):
    rand_rotation_dataset = []
    rows, cols = dataset[0].shape[: 2]
    for i in range(len(dataset)):
        img = dataset[i]
        random_angle = 30 * np.random.rand() - 15
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), random_angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        rand_rotation_dataset.append(dst)
    return rand_rotation_dataset


def rand_zoom(dataset):
    rand_zoom_dataset = []
    rows, cols = dataset[0].shape[: 2]
    for i in range(len(dataset)):
        img = dataset[i]
        px = np.random.randint(-2, 2)
        pts1 = np.float32([[px, px], [rows - px, px], [px, cols - px], [rows - px, cols - px]])
        pts2 = np.float32([[0, 0], [rows, 0], [0, cols], [rows, cols]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (rows, cols))
        rand_zoom_dataset.append(dst)
    return rand_zoom_dataset


def histogram_eq(dataset):
    hist_eq_dataset = []
    channels = np.shape(dataset[0])[2]
    for i in range(len(dataset)):
        img = dataset[i]
        for j in range(channels):
            img[:, :, j] = cv2.equalizeHist(img[:, :, j])
        hist_eq_dataset.append(img)
    return hist_eq_dataset


def normalize(dataset):
    return dataset / 128.0 - 1.0


def rgb_to_gray_scale(dataset):
    return np.sum(dataset / 3, axis=3, keepdims=True)






