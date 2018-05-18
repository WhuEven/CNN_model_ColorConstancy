# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:49:57 2018

@author: phamh
"""
import numpy as np;
import cv2;
import scipy.io;
from glob import glob;
from random import randint;
from progress_timer import progress_timer;

def generate_train_data(train_size, set_name, patch_size):
    
    #Load ground truth illum value
    if (set_name == 'Shi-Gehler'):
        mat_name = 'real_illum_568.mat';
        key = 'real_rgb';
        path = 'C:\\Users\\phamh\\Workspace\\Dataset\\Shi_Gehler\\Train_set\\';
        
    elif (set_name == 'Canon'):
        mat_name = 'Canon600D_gt.mat';
        key = 'groundtruth_illuminants';
        path = 'C:\\Users\\phamh\\Workspace\\Dataset\\Canon_600D\\Train_set\\';
        
    illum_mat = scipy.io.loadmat('GT_Illum_Mat\\' + mat_name, squeeze_me = True, struct_as_record = False);
    ground_truth_illum = illum_mat[key];
    
    flist = glob(path + '*.png');
    number_of_train_gt = len(flist);
    
    pt = progress_timer(n_iter = number_of_train_gt, description = 'Generating Training Data :');
    
    patches_per_image = int(train_size/number_of_train_gt);

    X_train_origin, Y_train_origin, name_train = [], [], [];
    i = 0;
    patch_r, patch_c = patch_size;

    while (i < number_of_train_gt):
        
        image_number = flist[i];
        index = (image_number.replace(path ,'')).replace('.png', '');
        
        image = cv2.imread(image_number);
        n_r, n_c, _ = np.shape(image);
        total_patch = int(((n_r - n_r%patch_r)/patch_r)*((n_c - n_c%patch_c)/patch_c));
        
        img_resize = cv2.resize(image, ((n_r - n_r%patch_r), (n_c - n_c%patch_c))); 
        img_reshape = np.reshape(img_resize, (int(patch_r), -1, 3));
        
        #Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8));
    
        for j in range (0, patches_per_image):
            
            rd = randint(0, total_patch - 1);
            img_patch = img_reshape[0:patch_r, rd*patch_c:(rd+1)*patch_c];
            
            #Convert image to Lab to perform contrast normalizing
            lab= cv2.cvtColor(img_patch, cv2.COLOR_BGR2LAB);
            
            #Contrast normalizing(Stretching)
            l, a, b = cv2.split(lab);
            cl = clahe.apply(l);
            clab = cv2.merge((cl, a, b));
            
            #Convert back to BGR
            img_patch = cv2.cvtColor(clab, cv2.COLOR_LAB2BGR);
            
            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB);
            
            X_train_origin.append(img_patch);
            Y_train_origin.append(ground_truth_illum[int(index) - 1]);
        
        name_train.append('%04d' % (int(index) - 1));
             
        i += 1;
        
        pt.update();
 
    X_train_origin = np.asarray(X_train_origin);
    Y_train_origin = np.asarray(Y_train_origin);
    
    X_train_origin = X_train_origin/255;
    max_Y = np.amax(Y_train_origin, 1);
    Y_train_origin[:, 0] = Y_train_origin[:, 0]/max_Y;
    Y_train_origin[:, 1] = Y_train_origin[:, 1]/max_Y;
    Y_train_origin[:, 2] = Y_train_origin[:, 2]/max_Y;
    
    seed = randint(1, 5000);
    np.random.seed(seed);
    X_train_origin = np.random.permutation(X_train_origin);
    
    np.random.seed(seed);
    Y_train_origin = np.random.permutation(Y_train_origin);
    
    pt.finish();
    
    return X_train_origin, Y_train_origin, name_train;

def generate_test_data(test_size, set_name, patch_size):
    
    #Load ground truth illum value
    if (set_name == 'Shi-Gehler'):
        mat_name = 'real_illum_568.mat';
        key = 'real_rgb';
        path = 'C:\\Users\\phamh\\Workspace\\Dataset\\Shi_Gehler\\Test_set\\';
        
    elif (set_name == 'Canon'):
        mat_name = 'Canon600D_gt.mat';
        key = 'groundtruth_illuminants';
        path = 'C:\\Users\\phamh\\Workspace\\Dataset\\Canon_600D\\Test_set\\'
        
    illum_mat = scipy.io.loadmat('GT_Illum_Mat\\' + mat_name, squeeze_me = True, struct_as_record = False);
    ground_truth_illum = illum_mat[key];
    
    flist = glob(path + '*.png');
    
    number_of_test_gt = len(flist);
    
    pt = progress_timer(n_iter = number_of_test_gt, description = 'Generating Testing Data :');
    
    patches_per_image = int(test_size/number_of_test_gt);
    X_test, Y_test, name_test = [], [], [];
    i = 0;
    
    patch_r, patch_c = patch_size;

    while (i < number_of_test_gt):
    
        image_number = flist[i];
        index = (image_number.replace(path ,'')).replace('.png', '');
        
        image = cv2.imread(image_number);
        n_r, n_c, _ = np.shape(image);
        total_patch = int(((n_r - n_r%patch_r)/patch_r)*((n_c - n_c%patch_c)/patch_c));
        
        img_resize = cv2.resize(image, ((n_r - n_r%patch_r), (n_c - n_c%patch_c))); 
        img_reshape = np.reshape(img_resize, (int(patch_r), -1, 3));
        
        #Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8));
        
        for j in range (0, patches_per_image):
            
            rd = randint(0, total_patch - 1);
            img_patch = img_reshape[0:patch_r, rd*patch_c:(rd+1)*patch_c];
            
            #Convert image to Lab to perform contrast normalizing
            lab= cv2.cvtColor(img_patch, cv2.COLOR_BGR2LAB);
            
            #Contrast normalizing(Stretching)
            l, a, b = cv2.split(lab);
            cl = clahe.apply(l);
            clab = cv2.merge((cl, a, b));
            
            #Convert back to BGR
            img_patch = cv2.cvtColor(clab, cv2.COLOR_LAB2BGR);
            
            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB);
            
            X_test.append(img_patch);
            Y_test.append(ground_truth_illum[int(index) - 1]);
        
        name_test.append('%04d' % (int(index) - 1));
             
        i += 1;
        pt.update();
 
    X_test = np.asarray(X_test);
    Y_test = np.asarray(Y_test);
    
    X_test = X_test/255;
    max_Y = np.amax(Y_test, 1);
    Y_test[:, 0] = Y_test[:, 0]/max_Y;
    Y_test[:, 1] = Y_test[:, 1]/max_Y;
    Y_test[:, 2] = Y_test[:, 2]/max_Y;
    
    seed = randint(1, 5000);
    np.random.seed(seed);
    X_test = np.random.permutation(X_test);
    
    np.random.seed(seed);
    Y_test = np.random.permutation(Y_test);
    
    pt.finish();
    
    return X_test, Y_test, name_test;

#train_size = 3000;
#test_size = 2420; 
#patch_size = (32, 32);
#set_name = 'Shi-Gehler';
#
#X_train, Y_train, name_train = generate_train_data(train_size, set_name, patch_size);
#np.save('X_train.npy', X_train); np.save('Y_train.npy', Y_train); np.save('name_train.npy', name_train);
#
#X_test, Y_test, name_test = generate_test_data(test_size, set_name, patch_size);
#np.save('X_test.npy', X_test); np.save('Y_test.npy', Y_test); np.save('name_test.npy', name_test);