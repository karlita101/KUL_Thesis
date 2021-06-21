## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color+ Adaptation by KARLA RIVERA               ##
#####################################################

# First import the libraries
import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import os
import keyboard
from imutils import perspective

def bit_tranf(rgb):
    #input rgb 8-bit
    #16bitColor = ((8bitColor + 1)*256) - 1 
    rgbd_16=((rgb+1)*256)-1
    rgbd_16=np.array(rgbd_16,np.uint16)
    return rgbd_16


def f(x):
    val_min=0
    val_max=255
    b=0
    dwork=1.5*1000
    return ((val_max-val_min)/(dwork-0)*x) + b


#Specify Data OUTPUT path
"""
path_rgb = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/RGBDBackground/RGB'
path_rgba = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/RGBDBackground/RGBA'
path_mask = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/RGBDBackground/Mask'

"""
"""
path_rgb = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/RGBDBackgroundtest/ARUCO/RGB'
path_rgba = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/RGBDBackgroundtest/ARUCO/RGBA'
path_mask = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/RGBDBackgroundtest/ARUCO/Mask'
"""
"""
path_rgb = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/RGBDBackgroundtest/NOpoly/RGB'
path_rgba = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/RGBDBackgroundtest/NOpoly/RGBA'
path_mask = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/RGBDBackgroundtest/NOpoly/Mask'
"""

#Get train data
"""
path_rgb = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/210408PhantomTrain/RGB'
path_rgba = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/210408PhantomTrain/RGBA'
path_mask = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/210408PhantomTrain/Mask'
path_depth = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/210408PhantomTrain/Depth'
path_TRUEdepth = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/210408PhantomTrain/TRUEDepth'
"""

#initialize test
path_rgb_test = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/210408PhantomTest/WOUT/RGB'
path_rgba_test = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/210408PhantomTest/WOUT/RGBA'
path_mask_test = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/210408PhantomTest/WOUT/Mask'
path_depth_test= 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/210408PhantomTest/WOUT/Depth'
path_TRUEdepth_test= 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/210408PhantomTest/WOUT/TRUEDepth'


#Get Camera Parameters
path=r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\calibration.txt'

#Load Intrinsics
param = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
camera_matrix = param.getNode("K").mat()
dist_coef = param.getNode("D").mat()

#Calll aruco dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)

#Initialize aruco IDs used for polygon shape: top left bottom left, top right and  bottom right
#arucoIDs=[2,35,100,200]
#21040 KR: add aruco ID's for test
arucoIDs = [4, 3, 300, 400]

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

#Create a config and configure the pipeline to stream different resolutions of color and depth streams
config = rs.config()
# 16 bit linear depth values
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 8 bit bgr

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

#210409 KR: Turn ON Emitter
depth_sensor.set_option(rs.option.emitter_always_on, 1)


# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

count = 0
count2=0

#initialize empty polygon corners
poly_corners = [None]*4

# Streaming loop
try:
    while True:

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        #D-16bit, RGB 8bit
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
             
        #Detect ARUCO Marker
        #Grayscale IMG
        gray_image=cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)

       
        #Detect Aruco 
        #Draw detected markers on RGB image
        arucoParameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, aruco_dict, parameters=arucoParameters, cameraMatrix=camera_matrix, distCoeff=dist_coef)
        
        #Don't show  ID's so that the DL  networks doesn't learn these
        #color_image=aruco.drawDetectedMarkers(color_image, corners,ids)
        #210409 KR: #DON'T DRAW!
        #aruco.drawDetectedMarkers(color_image, corners)
        corn_sq = np.squeeze(corners)
        #print(corn_sq)
        
        if ids is not None:    
           if len(corners) == 4:
                for i, id in enumerate(ids):
                    #print('ID index', i, 'Value',id)
                    if id == arucoIDs[0]:
                        poly_corners[0] = corn_sq[i, 0, :]
                    elif id == arucoIDs[1]:
                        poly_corners[1] = corn_sq[i, 1, :]
                    elif id == arucoIDs[2]:
                        poly_corners[2] = corn_sq[i, 2, :]
                    elif id == arucoIDs[3]:
                        poly_corners[3] = corn_sq[i, 3, :]
                    else:
                        print('Another ID was detected', id)
                #Draw the polyline border
                pts = np.array([poly_corners],np.int32)
                
                #KR 03/31: comment out to capture images without the polygon lines. Can  it  detect it
                #210409: DON'T DRAW BORDER
                #cv2.polylines(color_image, np.array([pts]), True, (0,0,255), 5)
                
                #Create a binary mask ( 1 channel)
                binary_mask=np.zeros((gray_image.shape),np.uint8)
                cv2.fillPoly(binary_mask, [pts], (255, 255, 255),8)

                #Only take the depth and  RGB selection inside the binary mask 
                # (ignore the rest since we only care about the  relative depth of the object surface)
                """
                depth_selection = cv2.bitwise_and(depth_image, depth_image, mask=binary_mask)
                color_selection=cv2.bitwise_and(color_image, color_image, mask=binary_mask)
                color_selection[binary_mask==0]=255
                """
           else:
                pts=None    
        else:
            #if there are no id's ( no aruco) detected pass NONE
            pts=None
                
        #Get depth image interms of  milimeters 
        #depth_true=depth_selection*depth_scale*1000 to get in mm
        depth_true = depth_image*depth_scale*1000
         
        """       
        #Scale  pixels  from 0-255. Where 255 corresponds  to the max working distance
        # d>dwork is set to 0
        norm_depth=f(depth_true)
        norm_depth=np.uint8(norm_depth)
        norm_depth[norm_depth>255]=0
        """
        
        """
        # Render images:
        #   depth align to color on left. Depth on right
        #Show the full color image
        depth_colormap = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))
        #images = np.hstack((color_image, depth_image)) 
        """
        
        #First create the image with alpha channel (selection)
        #rgba = cv2.cvtColor(color_image, cv2.COLOR_RGB2RGBA)

        #Then assign the mask to the last channel of the image
        #rgba[:, :, 3] = norm_depth
        
        """ EDIT 04-17"""
        #rgbd_16 dtype uint16
        rgbd_16=cv2.cvtColor(bit_tranf(color_image), cv2.COLOR_RGB2RGBA)
        rgbd_16[:, :, 3]=depth_image
        rgbd_16 = np.asanyarray(rgbd_16,dtype=np.uint16)
        #print("data type",type(rgbd_16))
        
        
        """Initialize Frames"""
        """
        cv2.namedWindow('Aligned RGB-D Frames', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Aligned RGB-D Frames',images)
        key = cv2.waitKey(1)       
        """
        
        if pts is not None:
        #Enter to begin capturing images
            if keyboard.is_pressed('Enter'):
                count += 1
                
                cv2.imwrite(os.path.join(path_rgb, str(count).zfill(4)+'.png'), color_image)
                #cv2.imwrite(os.path.join(path_rgba, str(count).zfill(4)+'.png'), rgba)
                cv2.imwrite(os.path.join(path_rgba, str(count).zfill(4)+'.png'), rgbd_16)
                cv2.imwrite(os.path.join(path_mask, str(count).zfill(4)+'.png'), binary_mask)
                cv2.imwrite(os.path.join(path_depth, str(count).zfill(4)+'.png'), norm_depth)
                cv2.imwrite(os.path.join(path_TRUEdepth, str(count).zfill(4)+'.png'), depth_image)
                print("Total images captured:", count)
                continue
            elif key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
        else:
            if keyboard.is_pressed('Enter'):
                count2 += 1
                print("Number of undetected ARUCO frames",count2)
                #210408: Capturing images with ARUCO, so if no polygon is detected then we ignore
                cv2.imwrite(os.path.join(path_rgb_test, str(count).zfill(4)+'.png'), color_image)
                #cv2.imwrite(os.path.join(path_rgba_test, str(count).zfill(4)+'.png'), rgba)
                cv2.imwrite(os.path.join(path_rgba_test, str(count).zfill(4)+'.png'), rgbd_16)
                cv2.imwrite(os.path.join(path_depth_test, str(count).zfill(4)+'.png'), norm_depth)
                cv2.imwrite(os.path.join(path_TRUEdepth_test, str(count).zfill(4)+'.png'), depth_image)
                #print("Total images captured:", count)
                
                continue
            elif key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
                    
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
