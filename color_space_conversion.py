# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:30:04 2018

@author: phamh
"""
import numpy as np;

def rgb2xyz(B_channel, G_channel, R_channel, color_space):
    B_channel = B_channel/255; G_channel = G_channel/255; R_channel = R_channel/255;
    if color_space == 'AdobeRGB':
        X = 0.5767*R_channel + 0.1856*G_channel + 0.1882*B_channel;
        Y = 0.2974*R_channel + 0.6273*G_channel + 0.0753*B_channel;
        Z = 0.0270*R_channel + 0.0707*G_channel + 0.9911*B_channel;
        return (X, Y, Z);
    elif color_space == 'sRGB':
        X = 0.4124*R_channel + 0.3576*G_channel + 0.1805*B_channel;
        Y = 0.2126*R_channel + 0.7152*G_channel + 0.0722*B_channel;
        Z = 0.0193*R_channel + 0.1192*G_channel + 0.9505*B_channel;
        return (X, Y, Z);
    
def xyz2rgb(X, Y, Z, color_space):
    if color_space == 'AdobeRGB':
        B_channel = 0.0134*X + -0.1184*Y + 1.0154*Z;
        G_channel = -0.9693*X + 1.8760*Y + 0.0416*Z;
        R_channel = 2.0414*X + -0.5649*Y + -0.3447*Z;
        B_channel = B_channel*255; G_channel = G_channel*255; R_channel = R_channel*255;
        return (B_channel, G_channel, R_channel);
    elif color_space == 'sRGB':
        B_channel = 0.0556*X + -0.2040*Y + 1.0572*Z;
        G_channel = -0.9693*X + 1.8760*Y + 0.0416*Z;
        R_channel = 3.2405*X + -1.5371*Y + -0.4985*Z;
        B_channel = B_channel*255; G_channel = G_channel*255; R_channel = R_channel*255;
        return (B_channel, G_channel, R_channel);

def f_function(i, i_ref):
    r = i/i_ref;
    m, n = r.shape;
    f = np.zeros((m, n));
    x, y = np.where(r > 0.008856);
    f[x, y] = r[x, y]**1/3;
    x, y = np.where(r <= 0.008856);
    f[x, y] = (7.787*r[x, y] + 16/116);  
    return f;
    
def xyz2lab(X, Y, Z, illuminant):
    if illuminant == 'D65':
        Xn = 95.047; 
        Yn = 100;
        Zn = 108.883;
    elif illuminant == 'D50':
        Xn = 96.6797; 
        Yn = 100;
        Zn = 82.5188;
        
    L = 116*f_function(Y, Yn) - 16;
    a = 500*f_function(X, Xn) + -500*f_function(Y, Yn);
    b = 200*f_function(Y, Yn) + -200*f_function(Z, Zn);
    
    return (L, a, b);

def lab2xyz(L, a, b, illuminant):
    if illuminant == 'D65':
        Xn = 95.047; 
        Yn = 100;
        Zn = 108.883;
    elif illuminant == 'D50':
        Xn = 96.6797; 
        Yn = 100;
        Zn = 82.5188;
        
    if L > 7.9996:
        X = Xn*((L/116 + a/500 + 16/116)**3);
        Y = Yn*((L/116 + 16/116)**3);
        Z = Zn*((L/116 - b/200 + 16/116)**3);
    elif L <= 7.9996:
        X = Xn*(1/7.787)*(L/116 + a/500);
        Y = Yn*(1/7.787)*(L/116);
        Z = Zn*(1/7.787)*(L/116 - b/200);
        
    return (X, Y, Z);
    
def rgb2lab(B_channel, G_channel, R_channel, color_space, illuminant):
    X, Y, Z = rgb2xyz(B_channel, G_channel, R_channel, color_space);
    L, a, b = xyz2lab(X, Y, Z, illuminant);
    return (L, a, b);

def lab2rgb(L, a, b, color_space, illuminant):
    X, Y, Z = lab2xyz(L, a, b, illuminant);
    B_channel, G_channel, R_channel = xyz2rgb(X, Y, Z, color_space);
    return (B_channel, G_channel, R_channel);