'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from scipy.ndimage.interpolation import rotate, shift, zoom

def rand_jitter(temp):
	if np.random.random() > .7:
		temp[np.random.randint(0,28,1), :] = 0
	if np.random.random() > .7:
		temp[:, np.random.randint(0,28,1)] = 0
	if np.random.random() > .7:
		temp = shift(temp, shift=(np.random.randint(-3,3,2)))
	if np.random.random() > .7:
		temp = rotate(temp, angle = np.random.randint(-20,20,1), reshape=False)
	return temp


batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


#plt.imshow(X_train[1000, :, :, 0].reshape(img_rows, img_cols).T, origin="lower")
#plt.show()


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# Callback for model saving:
checkpointer = ModelCheckpoint(filepath="auto_save_weights.hdf5", 
                                   verbose=1, save_best_only=True)

for j in range(nb_epoch):
	X_train_temp = np.copy(X_train)
	# Add noise on later epochs
	if j > 0:
		for k in range(0, X_train_temp.shape[0]):
			X_train_temp[k,:,:,0] = rand_jitter(X_train_temp[k,:,:,0])

	model.fit(X_train_temp, Y_train, batch_size=batch_size, nb_epoch=1,
			verbose=1, validation_data=(X_test, Y_test), callbacks = [checkpointer])

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
