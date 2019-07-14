# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:34:51 2018

@author: phamh
"""

import numpy as np
import cv2
import os

from create_local_illum_map import create_local_illum_map
#from sklearn.svm import SVR


def pool_function(illum_map, pool_size, stride,  mode):
    
    n_h, n_w, n_c = illum_map.shape
    pool_h, pool_w = pool_size
    stride_h, stride_w = stride
    
    # Define the dimensions of the output
    h_out = int(1 + (n_h - pool_h) / stride_h)
    w_out = int(1 + (n_w - pool_w) / stride_w)
    c_out = n_c
    
    pool_out = np.zeros((h_out, w_out, c_out))
    
    for h in range(h_out):                     
        for w in range(w_out):                 
            for c in range (c_out):            
                
                vert_start = h*stride_h
                vert_end = h*stride_h + pool_h
                horiz_start = w*stride_w
                horiz_end = w*stride_w + pool_w
                    
                slice_window = illum_map[vert_start:vert_end, horiz_start:horiz_end, c]
                    
                if mode == "std":
                    pool_out[h, w, c] = np.std(slice_window)
                elif mode == "average":
                    pool_out[h, w, c] = np.mean(slice_window)
                    
    return pool_out

def local_2_global(image_name, image, patch_size):

    n_r, n_c, _ = image.shape
    
    illum_map = create_local_illum_map(image, patch_size)      #patch_size = (32, 32)
    
    #Gaussian Smooting with 5x5 kernel
    illum_map_smoothed = cv2.GaussianBlur(illum_map, (5,5), cv2.BORDER_DEFAULT)
    
    #Perform median pooling
    median_pooling = np.zeros((1, 3))
    median_pooling[0, 0] = np.median(illum_map_smoothed[:, :, 0])
    median_pooling[0, 1] = np.median(illum_map_smoothed[:, :, 1])
    median_pooling[0, 2] = np.median(illum_map_smoothed[:, :, 2])
        
    #Perform avarage pooling 
    n_r, n_c, _ = illum_map.shape
    pool_size = (int(n_r/3), int(n_c/3))
    stride = (int(n_r/3), int(n_c/3))
    average_pooling = pool_function(illum_map, pool_size, stride, mode = 'average')
    
    average_pooling[0, 0] = np.mean(average_pooling[:, :, 0])
    average_pooling[0, 1] = np.mean(average_pooling[:, :, 1])
    average_pooling[0, 2] = np.mean(average_pooling[:, :, 2])
    
    #reshape_average = np.reshape(average_pooling, (-1, ));
    #svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    #illum_global = svr_rbf.fit(np.reshape(median_pooling, (3, 1)), reshape_average).predict(median_pooling);
    
    #Switch to average pooling if you prefer: 
    #illum_global = average_pooling;
    illum_global = median_pooling
    
    return illum_global
    
    
    
