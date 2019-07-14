# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 18:28:36 2018

@author: phamh
"""

import numpy as np
import cv2
from gamma_linearization import gamma_decode, gamma_encode
from local_2_global import local_2_global

#image name correspond to patches's sub-folder
#Exp: image 0015 corresponds to patches/0015 folder

def white_balancing(img, image_name, patch_size):
    
    m, n, _ = img.shape
    B_channel, G_channel, R_channel = cv2.split(img)
    Color_space = 'AdobeRGB'
 
    #Undo Gamma Correction
    B_channel, G_channel, R_channel = gamma_decode(B_channel, G_channel, R_channel, Color_space)
    
    #Compute local illuminant map then aggregate to global illuminant
    illum_global = local_2_global(image_name, img, patch_size)
        
    #Gain of red and  blue channels
    alpha = np.max(illum_global)/illum_global[0, 2]
    beta = np.max(illum_global)/illum_global[0, 1]
    ceta = np.max(illum_global)/illum_global[0, 0]
        
    #Corrected Image
    B_cor = alpha*B_channel
    G_cor = beta*G_channel
    R_cor = ceta*R_channel
        
    #Gamma correction to display
    B_cor, G_cor, R_cor = gamma_encode(B_cor, G_cor, R_cor, Color_space)
    B_cor[B_cor > 255] = 255
    G_cor[G_cor > 255] = 255
    R_cor[R_cor > 255] = 255
        
    #Convert to uint8 to display
    B_cor = B_cor.astype(np.uint8)
    G_cor = G_cor.astype(np.uint8)
    R_cor = R_cor.astype(np.uint8)
    img_cor = cv2.merge((B_cor, G_cor, R_cor))
    
    return img_cor

img = cv2.imread('E:\\e_data\\1.tif')
image_name = '1'
patch_size = (32, 32)
img_white_balance = white_balancing(img, image_name, patch_size)
cv2.imwrite(image_name + '_cor.png', img_white_balance)