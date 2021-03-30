# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 19:14:27 2021

@author: karla
"""

import cv2
#import os,glob
import os
import numpy as np
import os.path

from os import listdir,makedirs
from os.path import splitext
from os.path import isfile,join
from glob import glob
from PIL import Image

dstpath = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/RGBDframes/RGBA'



files = os.listdir(dstpath)
os.chdir(dstpath)
for file in files:
    img = Image.open(file)
    image=np.array(img)
    length=len(image.shape)
    #print(image.shape)
    #print (img.size)
    #print("length",length)




testimg=Image.open("0000.png")
image_nd=np.array(testimg)
image_ndtest=np.expand_dims(image_nd[:,:,0],axis=2)
cv2.imshow('image',image_ndtest)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Done-worked

#before edit you were in different directory and you want  edit file which is in another directory , so you have to change your directory to the same path you wanted to change some of it's file 
"""files = os.listdir(dstpath)
os.chdir(dstpath)
for file in files:
    name,ext=file.split('.')
    edited_name=str(int(name)).zfill(4)
    os.rename(file,edited_name+'.'+ext)
    print(file)

"""

# Used to check the path for the "Missing ID's ( was an error on my part in the code)
"""
#files = os.listdir(dstpath)
#os.chdir(dstpath)
ids=[splitext(file)[0] for file in listdir(path) if not file.startswith('.')]
    
idx=ids[43]
mask_file=glob(path+idx+''+'.*')

"""