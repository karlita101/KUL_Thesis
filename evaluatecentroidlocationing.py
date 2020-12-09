#Author: Karla R. 
#Sept 2020-
#Python 3.7 

import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import itertools 
import pandas as pd
import openpyxl

from relativepose import *
 
 #source 1
 #'https://medium.com/@muralimahadev40/aruco-markers-usage-in-computer-vision-using-opencv-python-cbdcf6ff5172'
 #Source 2
 #'https://medium.com/@aliyasineser/aruco-marker-tracking-with-opencv-8cb844c26628'
 #Source 3
 #'https://docs.opencv.org/master/d9/d6a/group__aruco.html#gab9159aa69250d8d3642593e508cb6baa'
 

def combpairs(markerids):
    #Create indices for the marker ids
    #Output pairs of possible id marker combinations using the N choose K tool in itertools
    indices = [*range(len(markerids))]
    comb = list(itertools.combinations(range(len(markerids)), 2))
    #print(comb)
    
    #unpack each pair
    comb = [[*pairs] for pairs in comb]
    return comb
 

def stereoRGB_depth(corners):

    #Inputs:
    
    #corners [[Nx4] and their corresponding pixels]
    #N is number of markers
    #Recall that order of corners are clockwise
    
    """Averaging"""
    #Get average x and y coordinate for all Nx4 corners
    print('corners',corners)
    #Ensure integer pixels
    x_center= [int(sum([i[0] for i in marker])/4) for N in corners for marker in N]
    y_center = [int(sum([i[1] for i in marker])/4) for N in corners for marker in N]
    center_pixels = list(zip(x_center, y_center))

    ### if too close will get an error because the mean c center 
    ### is being calculated outside of its boundaries... but that is weird because 
    #pose estimation finds the center without a proble
    return center_pixels

#This is INCORRECT##
"""def center_reprojection(color_image,id_tvec, id_rvec, camera_matrix, dist_coef):
    #Purpose is to see if a reporjection from world coordinate into 
    #image plane can locate the centroid of the aruco marker better
    
    #mg = cv2.imread('left12.jpg')
    h,  w = color_image.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coef,(w,h),1,(w,h))   
    
    #Undistort image
    dst = cv2.undistort(color_image, camera_matrix, dist_coef, None, newcameramtx)
    
    # Projection matrix P=K*RT https://docs.opencv.org/3.4/d0/daf/group__projection.html
    
    center_WORLD_image = []
    
    for i, center_coord in enumerate(id_tvec):
        ##result = cv2.projectPoints(center_coord, id_rvec[i], id_tvec[i], camera_matrix, dist_coef)
        R, __ = cv2.Rodrigues(id_rvec[i])
        T = center_coord.reshape(3,1)
        RT = np.c_[R, T]
        world = np.append(T,1)
        result = np.matmul(newcameramtx.reshape(3,3),RT)
        result=np.matmul(result,world)
        center_WORLD_image.append(result)
    
    return center_WORLD_image"""

def evalDepth(center_pixels, depth_val, id_tvec):
    
    """Error caused due to centerpixels: Note Dec 7,2020"""
    """ The center pixels is outputing center pixels that are out of frame"""
    #Exception has occurred: IndexError index 516 is out of bounds for axis 0 with size 480
    
    ##INPUTS###
    #id_tvec ==> translation vectors for each marker (x,y,z)) * intersted in z only here
    #id_vec is (N,1,1,3)
    
    #depth_image is the depth frame data that is aligned to RGB frameset 
    
    #depth_val is the value calculated for the entire depth data set *depth_scale()
    #############
        
    #Get z-depth value from ARUCO RGB measurements 
    centerdepth_aruco=[[vect[2] for vect in marker] for N in id_tvec for marker in N]
    print('centerdepth',centerdepth_aruco)
   # Initialize empty lists
    #center_pixels = []
    centerdepth_val = []

    print('center pixels',center_pixels)

    for pixelset in center_pixels:
        #pixel[0] is x and [1] is y
        print('xpixel', pixelset[0])
        print('ypix',pixelset[1])
        #d=depth_val[pixelset[0]][pixelset[1]]
        #print(d)
        centerdepth_val.append(depth_val[pixelset[0]][pixelset[1]])
    #error_RGB_depth = [abs(i-j)/i*100 if i != 0 else None for i, j in zip(centerdepth_val, centerdepth_aruco)]
    error_RGB_depth = [(i-j)/i*100 if i != 0 else None for i, j in zip(centerdepth_val, centerdepth_aruco)]
    print('Error',error_RGB_depth)
    return error_RGB_depth


#Specify Data OUTPUT path
data_path = r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\Depth Estimation Eval 1'

#Get Camera Parameters
path=r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\calibration.txt'

#Load Intrinsics
param = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
camera_matrix = param.getNode("K").mat()
dist_coef = param.getNode("D").mat()

############################ Set up realsense pipeline #########################
# Create a pipeline
pipeline = rs.pipeline() 

#Create a config and configure the pipeline to stream different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) #16 bit linear depth values
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #8 bit bgr

# Start streaming
profile = pipeline.start(config)

#Get Depth Scale in METERS
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Instruct alignment Depth to RBG
align_to = rs.stream.color
align = rs.align(align_to)  

#Calll aruco dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)

#Create an empty output storage for maker ids and tvectors
output_data=[]

################# Streaming loop###############
try:
    while True:
        # Access canera streams/framesets: RBG and depth
        frames = pipeline.wait_for_frames()

        # Run Alignment Process on accessed framesets
        aligned_frames = align.process(frames)

        # Get aligned frames (modified depth frame)
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        
        #Convert depth and RGB frames to numpy array
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        print('shape of depth image',depth_image.shape)
        color_image = np.asanyarray(color_frame.get_data())
        
        #Get Depth Values (implemented Dec 1/2020)
        #### CHECK #################################################
        #### NOT SURE WHAT THE DIFFERENCE IS WITH THIS VS .get_distance ( pixel x, y)
        depth_val = depth_image*depth_scale
        
        # Find RGB's corresponding Grayscale values(ARUCO requires grayscale for the threshold operations) 
        gray_image=cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
        
        #Detect Aruco 
        #Draw detected markers on RGB image
        arucoParameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, aruco_dict, parameters=arucoParameters, cameraMatrix=camera_matrix, distCoeff=dist_coef)
        color_image=aruco.drawDetectedMarkers(color_image, corners,ids)
        #ids : ndarray
        #corners: list
        print("Read ids",ids)
        
        
        ################# TESTING!!############
        """Find Corner Centers using AVERAGING"""
        
        print("Shape of corners", np.shape(corners))
        print('corners', corners)
        center_pixels = stereoRGB_depth(corners)
        
        
        #Estimate ARUCO Pose
        markerlen=3.19/100  #Square dimension [m]
        axis_len=0.01  #length of axis (select)
        
        #Test whether all array elements along a given axis evaluate to True.
        if np.all(ids != None):
            #Create empty lists to store read id values
            id_rvec=[]
            id_tvec=[]
            id_hold=[]

            #For Identified Markers find POSE estimation:
            for i in range(0,len(ids)):
                #Get translation and rotation vectors (1,1,3 shape)
                print(i)
                print('Corresponding ID value is', ids[i])
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], markerlen,camera_matrix,dist_coef)
                #("ID marker", ids[i], "Z-dpeth",tvec[1,1,3])
                #print(rvec)
                print("tvec")
                print(tvec)
                
                #Store Rvec and Tvec  values
                id_rvec.append(rvec)
                id_tvec.append(tvec)
                print('id_tvec shape',np.shape(id_tvec))
                print('id_tvec example',id_tvec)
                
                #Draw aruco pose axes
                aruco.drawAxis(color_image, camera_matrix, dist_coef, rvec, tvec, axis_len)  

        """Here we see that it is good enough!"""
         
        for c in center_pixels:
            cv2.circle(color_image, c, 5, (255, 0, 0), 2)
        
        
        ### RECTANGLE
        start_point = (0, 0)

        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        end_point = (640, 480)

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        image = cv2.rectangle(color_image, start_point, end_point, color, thickness)



        
        
        
        
        
        
        
        
         

          
        #Make depth image have similar [3 channel] structure as RGB
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) 

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #images = np.hstack((color_image, depth_colormap))


        cv2.namedWindow('Detect Aruco Markers', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Aruco Markers', color_image)
        key = cv2.waitKey(3)
        
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()

