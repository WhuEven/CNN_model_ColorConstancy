# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 13:36:15 2018

@author: phamh
"""

import numpy as np;
import h5py;
import tensorflow as tf;

from tensorflow.python.keras.applications.vgg16 import VGG16

from tensorflow.python.keras.layers import Input, Conv2D, Dropout, Activation, MaxPooling2D, Dense, Lambda;
from tensorflow.python.keras import backend as K
from SqueezeNet_simple_bypass import SqueezeNet, extract_layer_from_model;
from tensorflow.python.keras.models import Model, model_from_json, save_model, load_model;
from tensorflow.python.keras.optimizers import Adam;
from generate_data import generate_data;

SEPERATE_CONFIDENCE = False;

def FCN(input_shape):
    
    
    vgg16_model = VGG16(weights = 'imagenet', include_top = False, input_shape = input_shape);
    
    #Sq_net = squeezenet(float(input_shape));
    fire8 = extract_layer_from_model(vgg16_model, layer_name = 'block4_pool');
    
    pool8 = MaxPooling2D((3,3), strides = (2,2), name = 'pool8')(fire8.output);
    
    print(pool8.get_shape());
    
    fc1 = Conv2D(64, (6,6), strides= (1, 1), padding = 'same', name = 'fc1')(pool8);
    
    fc1 = Dropout(rate = 0.5)(fc1);
    
    print(fc1.get_shape());
    
    if SEPERATE_CONFIDENCE:
        fc2 = Conv2D(4 , (1, 1), strides = (1, 1), padding = 'same', activation = 'relu', name = 'fc2')(fc1);
        rgb = K.l2_normalize(fc2[:, :, :, 0:3], axis = 3);
        w, h = map(int, fc2.get_shape()[1:3]);
        
        confidence = fc2[:, :, :, 3:4];
        confidence = np.reshape(confidence, [-1, w*h]);
        confidence = K.softmax(confidence);
        confidence = np.reshape(confidence, shape=[-1, w, h, 1]);
        
        fc2 = rgb * confidence;
        
    else:
        fc2 = Conv2D(3, (1, 1), strides = (1, 1), padding = 'same', name = 'fc2')(fc1);
    
    fc2 = Activation('relu')(fc2);
    print(fc2.get_shape());
    
    fc2 = Conv2D(3, (15, 15), padding = 'valid', name = 'fc_pooling')(fc2);
    
    
    def norm(fc2):
        
        fc2_norm = K.l2_normalize(fc2, axis = 3);
        illum_est = K.tf.reduce_sum(fc2_norm, axis = (1, 2));
        illum_est = K.l2_normalize(illum_est);
        
        return illum_est;
    
    #illum_est = Dense(3)(fc2);
    
    illum_est = Lambda(norm)(fc2);
    
    print(illum_est.get_shape());
    
    FCN_model = Model(inputs = vgg16_model.input, outputs = illum_est, name = 'FC4');
    
    return FCN_model;

'''
squeezenet_file = open('squeezenet.json', 'r');
loaded_squeezenet_json = squeezenet_file.read();
squeezenet_file.close();
squeezenet = model_from_json(loaded_squeezenet_json);
    
#load weights into squeezenet
squeezenet.load_weights("squeezenet.h5");
squeezenet.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['acc']);
'''
train_size = 500;
test_size = 500;
patch_size = (512, 512);
set_name = 'Shi-Gehler';


X_train_origin, Y_train_origin, X_train_norm, Y_train_norm, _, = generate_data(train_size, set_name, patch_size, job = 'Train');
                                                      
#X_test, Y_test, _, = generate_data(test_size, set_name, patch_size, job = 'Test');

fcn_model = FCN(input_shape = X_train_origin.shape[1:4]);

fcn_model.compile(optimizer = Adam(lr = 0.0002), loss = 'cosine_proximity', metrics = ['acc']);
fcn_model.fit(X_train_origin, Y_train_origin, validation_split = 0.2, epochs = 20, batch_size = 16);

save_model(fcn_model, 'fcn_model.h5');



