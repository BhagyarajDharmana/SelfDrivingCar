from SelfDrivingCar.TrafficSignAssistant import classifier_utils
from SelfDrivingCar.TrafficSignAssistant import model
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf

train, valid, test = classifier_utils.load_data(
    '/home/intence/SelfDrivingCar/SelfDrivingCar/TrafficSignAssistant/DataLib/DataSet')
x_train, y_train = train['features'], train['labels']
x_valid, y_valid = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']

n_train, n_valid, n_test, image_shape, num_classes = classifier_utils.data_summary(train, valid, test)

figs, axs = plt.subplots(4, 5, figsize=(16, 8))
figs.subplots_adjust(hspace=.2, wspace=.001)
axs = axs.ravel()
for i in range(20):
    index = random.randint(0, len(x_train))
    image = x_train[index]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(y_train[index])
figs.show()

train_class, train_counts = np.unique(y_train, return_counts=True)
valid_class, valid_counts = np.unique(y_valid, return_counts=True)
test_class, test_counts = np.unique(y_test, return_counts=True)
deficit_index = train_class[train_counts < 800]
deficit_count = 800 - train_counts[deficit_index]
n_augment = np.sum(deficit_count)
X_train_augment = []
y_train_augment = []
for i in range(len(deficit_index)):
    augment_data, augment_class = classifier_utils.aug_dataset(x_train, y_train, deficit_count[i], deficit_index[i])
    X_train_augment.extend(augment_data)
    y_train_augment.extend(augment_class)

print(len(X_train_augment))

X_train_augment = np.array(X_train_augment)
y_train_augment = np.array(y_train_augment)

X_train = np.concatenate((x_train, classifier_utils.rand_translation(
    classifier_utils.rand_rotation(classifier_utils.histogram_eq(classifier_utils.rand_zoom(X_train_augment))))),
                         axis=0)
y_train = np.concatenate((y_train, y_train_augment), axis=0)

print(np.shape(X_train))
print(np.shape(y_train))

# plot distribution of augmented dataset
train_class, train_counts = np.unique(y_train, return_counts=True)
plt.figure(figsize=(15, 5))
plt.bar(train_class, train_counts)
plt.grid()
plt.title("Augmented training Dataset : class vs count")
plt.xlabel("Class")
plt.ylabel("Number of images")
plt.show()

X_train = np.array(X_train)
X_valid = np.array(x_valid)
X_valid = classifier_utils.normalize(classifier_utils.rgb_to_gray_scale(X_valid))
X_train = classifier_utils.normalize(classifier_utils.rgb_to_gray_scale(X_train))

figs, axs = plt.subplots(4, 5, figsize=(16, 8))
figs.subplots_adjust(hspace=.4, wspace=.001)
axs = axs.ravel()
for i in range(20):
    index = random.randint(0, len(X_train) - 1)
    image = X_train[index].squeeze()
    axs[i].axis('off')
    axs[i].imshow(image, cmap='gray')
    axs[i].set_title(int(y_train[index]))

figs.show()

model.train(X_train, y_train, X_valid, y_valid,epochs=100)
