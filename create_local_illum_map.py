# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 14:52:45 2018

@author: phamh
"""

import numpy as np;
import h5py;
import cv2;

from glob import glob;
from tensorflow.python.keras.models import model_from_json;

def create_local_illum_map(image, patch_size):
    
    #load cc_model
    json_file = open('cc_model.json', 'r');
    loaded_model_json = json_file.read();
    json_file.close();
    cc_model = model_from_json(loaded_model_json);

    #load weights into cc_model
    cc_model.load_weights("cc_model.h5");
    cc_model.compile(optimizer = 'Adam', loss = 'cosine_proximity', metrics = ['acc']);
    
    n_r, n_c, _ = image.shape;
    patch_r, patch_c = patch_size;
    
    total_patch = int(((n_r - n_r%patch_r)/patch_r)*((n_c - n_c%patch_c)/patch_c));
    
    img_resize = cv2.resize(image, ((n_r - n_r%patch_r), (n_c - n_c%patch_c))); 
    img_reshape = np.reshape(img_resize, (int(patch_r), -1, 3));
    
    illum_estimate = [];
    
    for i in range(total_patch):
        
        img_patch = img_reshape[0:patch_r, i*patch_c:(i+1)*patch_c];
        
        patch_cvt = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB);
        patch_cvt = np.expand_dims(patch_cvt, axis=0);
        illum_estimate.append(cc_model.predict(patch_cvt));
     
    illum_estimate = np.asarray(illum_estimate);
    illum_map = np.reshape(illum_estimate, (int((n_r - n_r%patch_r)/patch_r), int((n_c - n_c%patch_c)/patch_c), 3));
    
    return illum_map;
        
