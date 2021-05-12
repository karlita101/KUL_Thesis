import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import os
import keyboard
from imutils import perspective


im_path=r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210408PhantomTrain\Depth\0001.png'

img=cv2.imread(im_path,cv2.IMREAD_UNCHANGED)

#8 bit depth

print(type(img))