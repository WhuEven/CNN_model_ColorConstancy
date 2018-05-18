# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:23:49 2018

@author: phamh
"""
import numpy as np;
import cv2;
from color_space_conversion import rgb2lab;

def angular_error(ground_truth_image, corrected_image, measurement_type):
    B_gt, G_gt, R_gt = cv2.split(ground_truth_image);
    B_cor, G_cor, R_cor = cv2.split(corrected_image);
    
    if measurement_type == 'mean':  
        e_gt = np.array([np.mean(B_gt), np.mean(G_gt), np.mean(R_gt)]);
        e_est = np.array([np.mean(B_cor), np.mean(G_cor), np.mean(R_cor)]);  
    
    elif measurement_type == 'median':
        e_gt = np.array([np.median(B_gt), np.median(G_gt), np.median(R_gt)]);
        e_est = np.array([np.median(B_cor), np.median(G_cor), np.median(R_cor)]);
        
    error_cos = np.dot(e_gt, e_est)/(np.linalg.norm(e_gt)*np.linalg.norm(e_est));
    e_angular = np.degrees(np.arccos(error_cos));
    return e_angular;

def Euclidean_distance(ground_truth_image, corrected_image):
    B_gt, G_gt, R_gt = cv2.split(ground_truth_image);
    B_cor, G_cor, R_cor = cv2.split(corrected_image);
    
    L_gt, a_gt, b_gt = rgb2lab(B_gt, G_gt, R_gt, 'AdobeRGB', 'D65');
    L_cor, a_cor, b_cor= rgb2lab(B_cor, G_cor, R_cor, 'AdobeRGB', 'D65');
    
    delta = np.sqrt(np.square(L_gt - L_cor) + np.square(a_gt - a_cor) + np.square(b_gt - b_cor));
    average_Euclidean_dist = np.mean(delta);

    return average_Euclidean_dist;
    