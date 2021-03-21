# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:14:14 2021
 Purpsoe of this code is to look into the ultrasound data type and structure provided by Ruixuan.
@author: karla
"""

import cv2 as cv
#import numpy as np
import os
import glob
from PIL import Image
#from pathlib import Path


img_path= 'Ruixuan_USimages/train/image'
mask_path='Ruixuan_USimages/train/label'
binary='Ruixuan_USimages/train/binarylabel'

"""
#Get shape for IMG= Has 120 images
count_img=0
os.chdir(img_path)
for file in glob.glob("*.png"):
    im=cv.imread(file)
    count_img+=1
    print(im.shape)
print(count_img)
"""

"""
#Get shape for IMG= Has 120 images
count_mask=0
os.chdir(mask_path)
for file in glob.glob("*.png"):
    msk=cv.imread(file)
    count_mask+=1
    print(msk.shape)
print(count_mask)
"""


"""
b_count_mask=0
os.chdir(binary)
for filename in glob.glob("*.png"):
    msk=cv.imread(filename)
    msk=cv.cvtColor(msk, cv.COLOR_BGR2GRAY)
    cv.imwrite(filename, msk) 
    b_count_mask+=1
    print(msk.shape)
print(b_count_mask)



test_mask=cv.imread(filename)
print(test_mask.shape)
"""



count_mask=0
os.chdir(binary)
for filename in glob.glob("*.png"):
    msk=cv.imread(filename)
    print(msk.shape)
print(count_mask)






"""
test_mask=cv.imread(file)
cv.imshow('mask',test_mask)
gray=cv.cvtColor(test_mask, cv.COLOR_BGR2GRAY)
cv.imshow('gray_mask',gray)
cv.waitKey(0)
cv.destroyAllWindows()"""