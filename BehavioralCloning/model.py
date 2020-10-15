import csv
import math
import cv2
import numpy as np
from sklearn.utils import shuffle
import matplotlib.image as mping
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

samples = []
with open('/home/intence/SelfDrivingCar/SelfDrivingCar/BehavioralCloning/DataLib/data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1:]
print('Filenames loaded...')
training_samples, validation_samples = train_test_split(samples, test_size=0.2)
data_path = "/home/intence/SelfDrivingCar/SelfDrivingCar/BehavioralCloning/DataLib/data/data/"


def generator(train_samples, batch_size=32):
    while 1:
        shuffle(train_samples)
        for offset in range(0, len(train_samples), batch_size):
            images = []
            angles = []
            batch_samples = train_samples[offset:offset + batch_size]
            for batch_sample in batch_samples:
                center_path = data_path + batch_sample[0]
                center_path = center_path.replace(' ', '')
                centre_img = mping.imread(center_path)
                left_path = data_path + batch_sample[1]
                left_path = left_path.replace(' ', '')
                left_img = mping.imread(left_path)
                right_path = data_path + batch_sample[2]
                right_path = right_path.replace(' ', '')
                right_img = mping.imread(right_path)

                correction = 0.2
                centre_angle = float(batch_sample[3])
                left_angle = centre_angle + correction
                right_angle = centre_angle - correction

                angles.extend([centre_angle, left_angle, right_angle])
                images.extend([centre_img[70:135, :], left_img[70:135, :], right_img[70:135, :]])
            aug_images = []
            aug_angles = []
            for image, angle in zip(images, angles):
                aug_images.append(image)
                aug_angles.append(angle)
                aug_images.append(cv2.flip(image, 1))
                aug_angles.append(angle * -1)

            x_train = np.array(aug_images)
            y_train = np.array(aug_angles)

            yield shuffle(x_train, y_train)


# compile and train the model using the generator function
train_generator = generator(training_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

row, col, ch = 65, 320, 3

# CNN architecture by NVIDIA, ref: https://arxiv.org/pdf/1604.07316v1.pdf
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(row, col, ch)))  # Normalization
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

print('Training...')

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=math.ceil(len(training_samples * 6) / 32),
                                     validation_data=validation_generator,
                                     nb_val_samples=math.ceil(len(validation_samples) / 32), nb_epoch=5, verbose=1)
# note, the total training samples are 6 times per epoch counting both original
# and flipped left, right and center images
model.save('model.h5')
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
