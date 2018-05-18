# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np;
import cv2;
from glob import glob;
import os;
from progress_timer import progress_timer;

def patch_sampling(img, patch_size, index):
    
    n_r, n_c, _ = np.shape(img);
    patch_r, patch_c = patch_size;
    total_patch = int(((n_r - n_r%patch_r)/patch_r)*((n_c - n_c%patch_c)/patch_c));
    
    img_resize = cv2.resize(img, ((n_r - n_r%patch_r), (n_c - n_c%patch_c))); 
    img_reshape = np.reshape(img_resize, (int(patch_r), int(patch_c*total_patch), 3));

    #Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8));
    
    for i in range(total_patch):
        img_patch = img_reshape[0:patch_r, i*patch_c:(i+1)*patch_c];
        
        #Convert image to Lab to perform contrast normalizing
        lab= cv2.cvtColor(img_patch, cv2.COLOR_BGR2LAB);
        l, a, b = cv2.split(lab);
        cl = clahe.apply(l);
        clab = cv2.merge((cl, a, b));
        
        img_patch = cv2.cvtColor(clab, cv2.COLOR_LAB2BGR);
        count = "%04d" % index;
        path_name = 'patches\\' + count + "\\%04d_" % index + "%04d.png" % (i+1);
        
        if not os.path.exists('patches\\' + count):
            os.makedirs('patches\\' + count);
            
        cv2.imwrite(path_name, img_patch);
        
    return 0;   

path = r'C:\Users\phamh\Workspace\Dataset\Shi_Gehler\Converted';
img_list = glob(path + "\*.png");
patch_size = (32, 32);

pt = progress_timer(n_iter = len(img_list), description = 'Preparing Patches :');

for i in range(len(img_list)):
    index = i + 1;
    img = cv2.imread(img_list[i], 1);
    a = patch_sampling(img, patch_size, index);
    
    pt.update();
    
pt.finish();