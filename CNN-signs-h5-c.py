import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow import keras
from sklearn.utils import shuffle
import h5py


train = h5py.File('X_train.h5', 'r')
X_train = train.get('dataset_1')

labs = h5py.File('labels_train.h5', 'r')
y_train=labs.get('dataset_1')

X_train=np.array(X_train)
y_train=np.array(y_train)
         
X_train=X_train/255
X_train, y_train= shuffle(X_train, y_train)

model = keras.models.Sequential([
    #keras.layers.Dropout(0.2),
    keras.layers.Conv2D(filters=200, kernel_size=3, activation='relu',strides=1,input_shape=(30,30,3)),
    #keras.layers.MaxPool2D(pool_size=2,strides=2),
    #keras.layers.Conv2D(filters=100, kernel_size=3, activation='relu',strides=1),
    #keras.layers.Conv2D(filters=100, kernel_size=3, activation='relu',strides=1),
    keras.layers.Flatten(),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation=tf.nn.relu),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(43, activation='softmax')
])

model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

c=np.random.choice(39209,5000,replace=False)
model.fit(X_train[c,:],y_train[c], batch_size=50,epochs=5,verbose=2,validation_split=0.1)

test = h5py.File('/X_test.h5', 'r')
X_test = test.get('dataset_1')

labs_test = h5py.File('/labels_test.h5', 'r')
y_test=labs_test.get('dataset_1')

X_test=np.array(X_test)
X_test=X_test/255
y_test=np.array(y_test)

model.evaluate(X_test,y_test,verbose=2)

test.close()
labs_test.close()
train.close()
labs.close()
