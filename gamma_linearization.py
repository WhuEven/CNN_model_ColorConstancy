# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 09:58:22 2018

@author: phamh
"""
import numpy as np;
from linear_nonlinear_transform import linear_transform, non_linear_transform;

def gamma_decode(B_gamma, G_gamma, R_gamma, decoding_type):
    B_gamma = B_gamma/255; G_gamma = G_gamma/255; R_gamma = R_gamma/255; 
    if decoding_type == 'AdobeRGB':
        #Adobe RGB
        gamma = 1/2.2;
        B_gamma_decode = 255*(B_gamma**(1/gamma)); 
        G_gamma_decode = 255*(G_gamma**(1/gamma));
        R_gamma_decode = 255*(R_gamma**(1/gamma));
        return (B_gamma_decode, G_gamma_decode, R_gamma_decode);
    elif decoding_type == 'sRGB':
        #sRGB
        gamma = 1/2.4;
        f = 0.055; s = 12.92; T = 0.04045;
        B_gamma_inverse = linear_transform(B_gamma, gamma, f, s, T);
        G_gamma_inverse = linear_transform(G_gamma, gamma, f, s, T);
        R_gamma_inverse = linear_transform(R_gamma, gamma, f, s, T);
        return (B_gamma_inverse, G_gamma_inverse, R_gamma_inverse);
    
def gamma_encode(B_channel, G_channel, R_channel, encoding_type):
    B_channel = B_channel/255; G_channel = G_channel/255; R_channel = R_channel/255;
    #Non linear encoding
    if encoding_type == 'AdobeRGB':
        #Adobe RGB
        gamma = 1/2.2;
        if np.all(B_channel <= 0):
            B_gamma_cor = (B_channel**(gamma + 0j));
            B_gamma_cor = 255*(abs(B_gamma_cor));
        else:
            B_gamma_cor = 255*(B_channel**gamma);
            
        if np.all(G_channel <= 0):
            G_gamma_cor = (G_channel**(gamma + 0j));
            G_gamma_cor = 255*(abs(G_gamma_cor));
        else:
            G_gamma_cor = 255*(G_channel**gamma);
            
        if np.all(R_channel <= 0):
            R_gamma_cor = (R_channel**(gamma + 0j));
            R_gamma_cor = 255*(abs(R_gamma_cor));
        else:
            R_gamma_cor = 255*(R_channel**gamma);
        
        return (B_gamma_cor, G_gamma_cor, R_gamma_cor);
    elif encoding_type == 'sRGB':
        #sRGB
        gamma = 1/2.4;
        f = 0.055; s = 12.92; t = 0.0031308;
        B_gamma_cor = non_linear_transform(B_channel, gamma, f, s, t);
        G_gamma_cor = non_linear_transform(G_channel, gamma, f, s, t);
        R_gamma_cor = non_linear_transform(R_channel, gamma, f, s, t);
        return (B_gamma_cor, G_gamma_cor, R_gamma_cor);
