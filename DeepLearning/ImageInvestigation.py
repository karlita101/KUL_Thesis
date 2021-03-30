#Diagnose dataset

import numpy as np
import cv2 as cv
from PIL import Image
import os.path


dstpath = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/RGBDframes/RGBA'

files = os.listdir(dstpath)
os.chdir(dstpath)
for file in files:
    img = Image.open(file)
    print('shape of PIL RGBA',img.size)
    image=np.array(img)
    print("numpy shappe",image.shape)
    length=len(image.shape)


rgba=cv.imread(file,cv.IMREAD_UNCHANGED)
im = cv.cvtColor(rgba, cv.COLOR_BGRA2RGBA)
print(img.s)
l=len(im.shape)
im2 = Image.fromarray(im, 'RGBA')




dif=im