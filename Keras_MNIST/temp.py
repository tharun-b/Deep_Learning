# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
# fix random seed for reproducibility
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
# %%
from keras.datasets import mnist
 
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

from matplotlib import pyplot as plt
plt.imshow(X_train[0])

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(X_train.shape)
print(y_train.shape)
print(y_train[:10])



# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)



# %% Model

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding= 'same', input_shape=(1,28,28)))

print(model.output_shape)

model.add(Conv2D(32, (3, 3), padding= 'same', activation='relu'))


model.add(MaxPooling2D(pool_size=(2,2), padding= 'same'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# %% Fitting

model.fit(X_train, Y_train, 
          batch_size=32, epochs=1, verbose=1)

# %% Testing

score = model.evaluate(X_test, Y_test, verbose=1)

pred = model.predict(X_test,  batch_size=None, verbose=1, steps=None)
predcat = np.argmax(pred, axis = 1)

cm= confusion_matrix(predcat,y_test)

plt.imshow(cm)


