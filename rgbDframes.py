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


def f(x):
    val_min=0
    val_max=255
    b=0
    dwork=1.5*1000
    return ((val_max-val_min)/(dwork-0)*x) + b





#Specify Data OUTPUT path
path_rgb = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/RGBDframes/RGB'
path_d = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/RGBDframes/Depth'
path_mask = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/RGBDframes/Mask'

#Get Camera Parameters
path=r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\calibration.txt'

#Load Intrinsics
param = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
camera_matrix = param.getNode("K").mat()
dist_coef = param.getNode("D").mat()

#Calll aruco dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)

#Initialize aruco IDs used for polygon shape: top left bottom left, top right and  bottom right
arucoIDs=[2,35,100,200]



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

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

count = 0
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

        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        
        ## TO AVOID BLACK DEPTH IMAGE REPLACE#
        """
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(
            aligned_depth_image), cv2.COLORMAP_RAINBOW)
        color_image = np.asanyarray(color_frame.get_data())
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        
        """
        
        #depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
           
     
       
        #Detect ARUCO Marker
        #Grayscale IMG
        gray_image=cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)

       
        #Detect Aruco 
        #Draw detected markers on RGB image
        arucoParameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, aruco_dict, parameters=arucoParameters, cameraMatrix=camera_matrix, distCoeff=dist_coef)
        color_image=aruco.drawDetectedMarkers(color_image, corners,ids)
        corn_sq = np.squeeze(corners)
        print(corn_sq)
        
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
                cv2.polylines(color_image, np.array([pts]), True, (0,0,255), 5)
                #Create a binary mask ( 1 channel)
                binary_mask=np.zeros((gray_image.shape),np.uint8)
                cv2.fillPoly(binary_mask, [pts], (255, 255, 255),8)
                #cv2.imshow('Binary Mask',binary_mask)
                #cv2.waitKey(1)


                #Only take the depth selection inside the binary mask (ignore the rest since we only care about the  relative depth of the object surface)
                depth_selection = cv2.bitwise_and(depth_image, depth_image, mask=binary_mask)
                #print('Depth Shape',depth_selection.shape)  #4800x640
                
                #Get depth image interms of  milimeters 
                depth_true=depth_selection*depth_scale*1000
                
                norm_depth=f(depth_true)
                norm_depth=np.uint8(norm_depth)
                norm_dept[norm_depth>255]=0
                
                """
                #normalize the image based on a subsection region of interest (ROI)
                #MUST  exclude 0  or else black will  dominate
                ROI_mean = np.mean(np.where(depth_selection > 0))
                ROI_std = np.std(np.where(depth_selection > 0))

                #Clip frame to lower and upper STD
                offset = 0.2
                clipped = np.clip(depth_selection, ROI_mean - offset*ROI_std, ROI_mean + offset*ROI_std).astype(np.uint8)

                # Normalize to range
                result = cv2.normalize(clipped, clipped, 0, 255, norm_type=cv2.NORM_MINMAX)
                """
                
                for pair in pts[0]:
                    print("Depth val", depth_image[pair[1],pair[0]])      
                
        """ Sanity Check
        for  marker in corn_sq:
            for corn in marker:
                cv2.circle(color_image, tuple(corn), 1, (0, 0, 255), 4)
        
        cv2.circle(color_image, tuple([228, 399]), 1, (0, 0, 255), 4)
        """
       
        # Render images:
        #   depth align to color on left 
        #   depth on right
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))
        #images = np.hstack((color_image, depth_image)) 
       
       

        """Show Frames"""
        cv2.namedWindow('Aligned RGB-D Frames', cv2.WINDOW_NORMAL)
        cv2.imshow('Aligned RGB-D Frames',images)
        key = cv2.waitKey(1)       
       
      
       
        #Enter to begin capturing images
        if keyboard.is_pressed('Enter'):
            count += 1
            
            cv2.imwrite(os.path.join(path_rgb, str(count)+'.png'), color_image)
            #grey level image file and hard to save detail. RS suggested making a change to depth_image
            #cv2.imwrite(os.path.join(path_d, str(count)+'.png'), depth_image)
            cv2.imwrite(os.path.join(path_mask, str(count)+'.png'), binary_mask)
            print("Images has been captured.")
            continue
                
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
