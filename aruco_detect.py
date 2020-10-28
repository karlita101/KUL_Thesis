import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
 
 
 #source 1
 #'https://medium.com/@muralimahadev40/aruco-markers-usage-in-computer-vision-using-opencv-python-cbdcf6ff5172'
 #Source 2
 #'https://medium.com/@aliyasineser/aruco-marker-tracking-with-opencv-8cb844c26628'
 #Source 3
 #'https://docs.opencv.org/master/d9/d6a/group__aruco.html#gab9159aa69250d8d3642593e508cb6baa'
 
 
 
 
#Get Camera Parameters
path=r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\calibration.txt'
#myLoadedData = cv2.Load(path)
param = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
camera_matrix = param.getNode("K").mat()
dist_coef = param.getNode("D").mat()
#print (fn.mat())

 
 
# Find and decode arucomarkers

# Create a pipeline
pipeline = rs.pipeline() 


#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) #16 bit linear depth values
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #8 bit bgr
    # Note: I thought that direct grayscale stream woould be useful  but it would be best to  only gray scale for identification and show rgb stream
    #config.enable_stream(rs.stream.color, 640, 480, rs.format.y16, 30) #16 per  pixel grayscale image

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
        gray_image=cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
        
 
        #ARUCO [Source 1,2,3]
        arucoParameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, aruco_dict, parameters=arucoParameters, cameraMatrix=camera_matrix, distCoeff=dist_coef)

        
        
        color_image=aruco.drawDetectedMarkers(color_image, corners,ids)
        
       
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