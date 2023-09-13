# -*- coding: utf-8 -*-
"""
Submission Template for Lab 8
Important notice: DO NOT use any global variables in this submission file
"""

# Task 1
def data_preprocessing(data_dir, cate2Idx, img_size):
  x = []
  y = []
  ###############################################################################
  # TODO: your code starts here
  for pokemon in sorted(os.listdir(data_dir)):
    label=cate2Idx[pokemon]
    print(label)
    for img in os.listdir(os.path.join(data_dir, pokemon)):
      # print(label)
      img_data=cv2.imread(os.path.join(data_dir, pokemon, img))
      img_data=cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
      resized_img_data=cv2.resize(img_data, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
      x.append(resized_img_data)
      y.append(label)
  # TODO: your code ends here
  ###############################################################################
  x = np.asarray(x)
  y = np.asarray(y)
  return x, y


# Task 2
def get_datagen():
  datagen = None
  ###############################################################################
  # TODO: your code starts here
  datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
  # TODO: your code ends here
  ###############################################################################
  return datagen


# Task 3
def custom_model():
  model = None
  ###############################################################################
  # TODO: your code starts here
  model = Sequential(
    [Conv2D(filters=512, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),  # Add a convolutional layer with 32 kernels, each of size 3x3
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=512, kernel_size=(3, 3), activation='relu'),                            # Add another convolutional layer with 64 kernels, each of size 3x3
    MaxPooling2D(pool_size=(2, 2)), 
    Conv2D(filters=512, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=512, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)), 
    Dropout(0.2),
    Flatten(),  
    Dense(units=900, activation='relu'),
    Dropout(0.2),
    Dense(units=400, activation='relu'), 
    Dropout(0.2),
    Dense(units=200, activation='relu'),                                                  # Add a dense layer (fully-connected layer) and use ReLU activation function
    Dropout(0.2),
    Dense(units=150, activation='softmax')]                                                # Add a dense layer (fully-connected layer) and use Softmax activation function
  )
  # TODO: your code ends here
  ###############################################################################
  return model


if __name__ == '__main__':
  # Import necessary libraries
  import os, cv2
  import numpy as np
  from sklearn.model_selection import train_test_split
  import keras
  from keras.utils import np_utils
  from keras.models import Sequential
  from keras.layers import Conv2D, MaxPooling2D
  from keras.layers import Dense, Dropout, Flatten
  from keras.preprocessing.image import ImageDataGenerator
