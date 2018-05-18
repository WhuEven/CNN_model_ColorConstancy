# -*- coding: utf-8 -*-
"""
Created on Wed May  2 11:44:06 2018

@author: phamh
"""

import numpy as np;
import cv2;

from error_measurement import angular_error;
from glob import glob;
from white_balancing import white_balancing;
from progress_timer import progress_timer;

casted_list = glob('Color-casted\\*.png');
corrected_list = glob('Corrected\\*.png');
gt_list = glob('Ground-truth\\*.png');
ang_error = [];
patch_size = (64, 64);

pt = progress_timer(n_iter = len(casted_list), description = 'Processing :');

for i in range (0, len(casted_list)):
    
    img = cv2.imread(casted_list[i]);
    image_name = casted_list[i].replace('Color-casted\\', '');
    image_name = image_name.replace('.png', '');
    
    img_gt = cv2.imread(gt_list[i]);
    
    img_cor = white_balancing(img, image_name, patch_size);
    cv2.imwrite('Corrected\\' + image_name + '_cor.png', img_cor);
    
    error = angular_error(img_gt, img_cor, 'mean');
    ang_error.append(error);
    
    pt.update();
    
avg_ang_error = np.mean(ang_error);
pt.finish();