# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 09:59:00 2018

@author: phamh
"""
import numpy as np;
import cv2;

def linear_transform(channel_sRGB, gamma, f, s, T):
    m, n = channel_sRGB.shape;
    channel_linear = channel_sRGB; channel_sRGB_norm = channel_sRGB/255;
    for i in range(0, m):
        for j in range(0, n):
            if (T < channel_sRGB_norm[i][j] < 1):
                channel_linear[i][j] = ((channel_sRGB_norm[i][j] + f)/(1 + f))**(1/gamma);
            elif (0 <= channel_sRGB_norm[i][j] <= T):
                channel_linear[i][j] = channel_sRGB_norm[i][j]/s;
                
    channel_linear = 255*channel_linear; 
    channel_linear[channel_linear > 255] = 255;            
    return channel_linear;

def non_linear_transform(channel, gamma, f, s, t):
    m, n = channel.shape;
    channel_gamma_cor = channel; channel_norm = channel/255;
    for i in range(0, m):
        for j in range(0, n):
            if (t < channel_norm[i][j] < 1):
                channel_gamma_cor[i][j] = (1 + f)*(channel_norm[i][j]**gamma) - f;
            elif (0 <= channel_norm[i][j] <= t):
                channel_gamma_cor[i][j] = s*channel_norm[i][j];
    
    channel_gamma_cor = channel_gamma_cor*255; 
    channel_gamma_cor[channel_gamma_cor > 255] = 255;       
    return channel_gamma_cor;
    