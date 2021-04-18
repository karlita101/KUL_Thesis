## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
# The Following code attempts to apply Point-cloud visualization using Open3d
# PC is constructed from RGB and Depth Frame 


#####################################################




import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import os
import keyboard
from imutils import perspective
from datetime import datetime
from open3d import *


def get_int(camera_matrix, H=480, W=640):
    #Need o3d input format
    #Made originally for RGB deoth 
    fx = camera_matrix[0, 0]
    cx = camera_matrix[0, 2]
    fy = camera_matrix[1, 1]
    cy = camera_matrix[1, 2]
    intrinsic = open3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    return intrinsic


#########################RGB Intrinsics######################################################
#Intrinsic Camera Path

path = r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\calibration.txt'
#Load Intrinsics
param = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
camera_matrix = param.getNode("K").mat()

#RGB Camera intrinsics (WRONG)
intrinsic_od3 = get_int(camera_matrix, H=480, W=640)
################################################################################################



pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Start streaming
profile = pipeline.start(config)
# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

#Turn ON Emitter
depth_sensor.set_option(rs.option.emitter_always_on, 1)

#Align
align_to = rs.stream.color
align = rs.align(align_to)

#frame rate counter
frate_count=[]


# Streaming loop
try:
    geometry_added = False
    #vis = Visualizer()
    vis = open3d.visualization.Visualizer()
    #vis.create_window('Aligned Column', width=640, height=480)
    vis.create_window("Test")
    pcd = open3d.geometry.PointCloud()
    
    
    while True:
        #initalize time
        dt0 = datetime.now()
        vis.add_geometry(pcd)
        #Add
        pcd.clear()
        
        # Get frameset of color and depth

        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        #Get Depth Intrinsics 
        #KR 0418 ignore dpeth intrinsics
        #Return pinhole camera intrinsics for Open3d
        #intrinsics = aligned_frames.profile.as_video_stream_profile().intrinsics
        ##pinhole_camera_intrinsic = open3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        #print(intrinsics.width, intrinsics.height)
        
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        #Alignment FPS
        process_time_alignment = datetime.now() - dt0
        print("Process Time Alignment",process_time_alignment)
        print("Alignment FPS = {0}".format(1/process_time_alignment.total_seconds()))

        if not aligned_depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        img_color = open3d.geometry.Image(color_image)
        img_depth = open3d.geometry.Image(depth_image)
        
        #Create RGBD image
        #rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, depth_scale*1000, depth_trunc=2000, convert_rgb_to_intensity=False)
        rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth,convert_rgb_to_intensity=False)

        ## KR 0417: USE DEPTH CAMERA INTRINSICS
        #KR 0418: reactivate
        #Create PC from RGBD
        temp= open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_od3)
        
        #KR 0418: Use color camera intrinsics since the depth frame was already aligned with the depth 
        #temp = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
        
        
        # Flip it, otherwise the pointcloud will be upside down
        temp.transform([[1, 0, 0, 0], [0, -1, 0, 0],
                       [0, 0, -1, 0], [0, 0, 0, 1]])

        pcd.points=temp.points
        pcd.colors=temp.colors

        
        #alignment FPS
        process_time_pc = datetime.now() - dt0
        print("Process Time PC", process_time_pc)
        print("Alignment FPS = {0}".format(
            1/process_time_pc.total_seconds()))
            
        vis.update_geometry(pcd)
        vis.poll_events()    
        vis.update_renderer()
        
        #Calculate process time for PC visualization
        process_time = datetime.now() - dt0
        print("FPS = {0}".format(1/process_time.total_seconds()))
        
        frate_count = np.append(frate_count, 1/process_time.total_seconds())
        #frame_count += 1
        
        # Press esc or 'q' to close the image window
        #if key & 0xFF == ord('q') or key == 27:
            #cv2.destroyAllWindows()
            #break

except KeyboardInterrupt:
    print("Press Ctrl-C to terminate while statement")
    mean_frp=np.mean(frate_count)
    std_frp=np.std(frate_count)
    print("Mean FRP PC estimation", mean_frp)
    print("Stdev FRP PC estimation", mean_frp)
    pass

finally:
    pipeline.stop()

vis.destroy_window()
