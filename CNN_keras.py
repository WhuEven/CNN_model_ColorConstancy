# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:27:20 2018

@author: phamh
"""
import h5py
import numpy as np
import cv2;
import tensorflow as tf;

from tensorflow.python.keras import layers, optimizers
from tensorflow.python.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.python.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.python.keras import backend as K
from generate_data import generate_train_data, generate_test_data;



def ColorNet(input_shape, channels = 3):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Stage 1
    X = Conv2D(240, (1, 1), strides = (1, 1), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(name = 'bn_conv1')(X)
    
    X = MaxPooling2D((8, 8), strides=(8, 8))(X)
  
    # output layer
    X = Flatten()(X)
    X = Dense(40, activation='relu', name='fc' + str(40))(X);
    X = Dropout(rate = 0.5)(X);
    
    X = Dense(channels, activation=None, name='fc' + str(channels))(X);
    
    # Create model
    color_model = Model(inputs = X_input, outputs = X, name='ColorNet');
    
    return color_model;

#X_train = np.load('X_train.npy'); Y_train = np.load('Y_train.npy');
#X_test = np.load('X_test.npy'); Y_test = np.load('Y_test.npy');
    
X_train, Y_train, _ = generate_train_data(train_size = 9000, set_name = 'Shi-Gehler', patch_size = (64, 64));
X_test, Y_test, _ = generate_test_data(test_size = 2420, set_name = 'Shi-Gehler', patch_size = (64, 64));

rmsprop = optimizers.RMSprop(lr = 0.001, rho=0.9, epsilon=None, decay=0.0);

cc_model = ColorNet(input_shape = X_train.shape[1:4]);
cc_model.compile(optimizer = 'Adam', loss = 'cosine_proximity', metrics = ['acc']);

estimate = cc_model.fit(X_train, Y_train, validation_split = 0.3333, epochs = 20, batch_size = 160);

preds = cc_model.evaluate(X_test, Y_test);
print();
print ("Loss = " + str(preds[0]));
print ("Test Accuracy = " + str(preds[1]));

# serialize model to JSON
model_json = cc_model.to_json()
with open("cc_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cc_model.save_weights("cc_model.h5")
print("Saved model to disk")
 

# =============================================================================
# img1 = cv2.imread('0044_0002.png');
# img1 = np.expand_dims(img1, axis=0);
# a = cc_model.predict(img1);
# 
# img2 = cv2.imread('0015_0015.png');
# img2 = np.expand_dims(img2, axis=0);
# b = cc_model.predict(img2);
# =============================================================================
