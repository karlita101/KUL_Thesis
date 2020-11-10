import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import itertools as iter

from relativepose import *
 
 #source 1
 #'https://medium.com/@muralimahadev40/aruco-markers-usage-in-computer-vision-using-opencv-python-cbdcf6ff5172'
 #Source 2
 #'https://medium.com/@aliyasineser/aruco-marker-tracking-with-opencv-8cb844c26628'
 #Source 3
 #'https://docs.opencv.org/master/d9/d6a/group__aruco.html#gab9159aa69250d8d3642593e508cb6baa'
 
def combpairs(markerids):
    #create indices for the marker ids
    #output pairs of possible id marker combinations using the N choose K tool in itertools 
    indices=[*range(len(markerids))]
    comb=list(itertools.combinations(range(len(markerids)), 2))
    print(comb)
    return comb


#Get Camera Parameters
path=r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\calibration.txt'
#myLoadedData = cv2.Load(path)
param = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
camera_matrix = param.getNode("K").mat()
dist_coef = param.getNode("D").mat()
#numpy.mat turns it into a matrix

# Create a pipeline
pipeline = rs.pipeline() 


#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) #16 bit linear depth values
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #8 bit bgr

# Start streaming
profile = pipeline.start(config)

# Create an align object depth and another stream
align_to = rs.stream.color
align = rs.align(align_to)  

#Calll aruco dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        # frames.get_depth_frame() is a 640x360 depth image
        frames = pipeline.wait_for_frames()


        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned framess
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        #Get depth and RGB Data, and convert RBG to grayscale
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #ARUCO asks for grayscale for the threshold operations
        gray_image=cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
        
 
        #Detect Aruco 
        arucoParameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, aruco_dict, parameters=arucoParameters, cameraMatrix=camera_matrix, distCoeff=dist_coef)
        color_image=aruco.drawDetectedMarkers(color_image, corners,ids)
        
        #ids : ndarray
        #corners: list
        print("ids",ids)
        print(corners)
        
        #Estimate Pose
        markerlen=3.19/100 #m
        axis_len=0.01
        
        #Test whether all array elements along a given axis evaluate to True.
        if np.all(ids != None):
            #Create empty lists to store values
            id_Rvec=[]
            id_tvec=[]
        
            for i in range(0,len(ids)):
                print(i)
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], markerlen,camera_matrix,dist_coef)
                print('ID marker', ids[i], 'Z-dpeth',tvec[i,1,3])
                #print(rvec)
                #print("tvec")
                #print(tvec)
                
                #Here we get rve and tvec as a (1,1,3) shape. Why?
                
                #Store Rvec and Tvec  values
                id_Rvec.append(rvec)
                id_tvec.append(tvec)
                #Draw  axes
                aruco.drawAxis(color_image, camera_matrix, dist_coef, rvec, tvec, axis_len)  

            #Relative pose: use comprehension lists
            comb=combpairs(ids)
            R_rel, t_rel =[relativePose(id_rvec[pairs[0]],id_rvec[pair[1]],id_tvec[pairs[0]],id_tvec[pairs[1]])  for pairs in comb]
            print('Rotation',R_rel,'for pairs', comb)
            print('translation',t_rel,'for pairs', comb)
            
            
        #make depth image of similar structure as color_images
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) 

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        ##images = np.hstack((color_image, depth_colormap))
        images = np.hstack((color_image, depth_colormap))


        cv2.namedWindow('Detect Aruco Markers', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Aruco Markers', images)
        key = cv2.waitKey(3)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()