from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
import pyrealsense2 as rs
 
def decode(im) :
    # Find barcodes and QR codes
    decodedObjects = pyzbar.decode(im)
    
    #Print results
    for obj in decodedObjects :
        print ('Type : ', obj.type)
        print ('Data : ', obj.data,'\n')
        
    return decodedObjects
    
   

# Display barcode and QR code location 
def display(im, decodedObjects):
    #Loop over all decoded objects    
    for decodedObject in decodedObjects:
        points = decodedObject.polygon
        
        #if points do not form  a quad, find a convex hull
        if len(points)> 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
            
        else:
            hull = points
        #number  of  points in the convex hull
        n=len(hull)
        
        #Draw the convext hull
        for j in range(0,n):
            cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)
     
     
    #Display Results
    #waitKey(0) will display the window infinitely until any keypress (it is suitable for image display).
    #waitKey(1) will display a frame for 1 ms, after which display will be automatically closed
    
    #cv2.imshow("Results",im)
    #cv2.waitKey(0)       
    return im
            
 


# Find and decode barcodes and QR codes

# Create a pipeline
pipeline = rs.pipeline() 

#pipeline which is a top level API for using RealSense depth cameras. 
#rs2::pipeline automatically chooses a camera from all connected cameras,
# so we can simply call pipeline::start() and the camera is configured and streaming:

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
##clipping_distance_in_meters = 1 #1 meter
##clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to) # K:  alignn color to depth


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
        #print("depth image: ", depth_image)
        color_image = np.asanyarray(color_frame.get_data())
        
        
        #print("color image: ", color_image)
        
        #Get objects
        ###decodedObjects=decode(im)
        decodedObjects=decode(color_image)
        #print ('decoded objects',decodedObjects)
        color_boxedframe=display(color_image, decodedObjects)
        ###color_boxedimage = np.asanyarray(color_boxedframe)
       
       ##05/10 color_image = np.asanyarray(color_frame.get_data())
        #print("color image: ", color_image)

        #KR: Find QR code
       ## 05/10 im=color_frame.get_data()
        ##Note sure if this works
        #im=np.asanyarray(color_frame.get_data())
       ## 05/10 decoded_objects=decode(im)
       ##05/10 boxedQR_im=display(im,decoded_objects)

        #make depth image of similar structure as color_images
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) 

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        ##images = np.hstack((color_image, depth_colormap))
        images = np.hstack((color_boxedframe, depth_colormap))


        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(3)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()