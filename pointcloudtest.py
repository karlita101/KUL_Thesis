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
import open3d as o3d
from open3d import *
import matplotlib.pyplot as plt

path_rgb = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/o3dtest/RGB'
path_depth = 'C:/Users/karla/OneDrive/Documents/GitHub/KUL_Thesis/o3dtest/Depth'


def get_int(camera_matrix,H=480,W=640):
    #Need o3d
    fx=camera_matrix[0,0]
    cx = camera_matrix[0, 2]
    fy = camera_matrix[1, 1]
    cy = camera_matrix[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    return intrinsic

#Get Camera Parameters
path=r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\calibration.txt'

#Load Intrinsics
param = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
camera_matrix = param.getNode("K").mat()
dist_coef = param.getNode("D").mat()

intrinsic_od3 = get_int(camera_matrix, H=480, W=640)


# Create a pipeline
pipeline = rs.pipeline()

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

#Turn ON Emitter
depth_sensor.set_option(rs.option.emitter_always_on, 1)


# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

count = 0


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
        
        #pc = rs.pointcloud()
        #pc.map_to(color_frame)
        #pointcloud = pc.calculate(aligned_depth_frame)
        #o3d.visualization.draw_geometries([pointcloud])
        
        """
        #https://github.com/dorodnic/binder_test/blob/master/pointcloud.ipynb
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        pointcloud = pc.calculate(aligned_depth_frame)
        print("Type point cloud is",type(pointcloud)
            
        
        
        #Worked but seems to take multiple frames. BUT TOO SLOW. Pointcloud took 12 frames
        pointcloud.export_to_ply("Test 1 .ply", color_frame)
        pcd = o3d.io.read_point_cloud("Test 1 .ply")
        print("pcd",pcd)
        print(np.asarray(pcd.points))
        o3d.visualization.draw_geometries([pcd])
        """


        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        #Imas
        #depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))
        
        cv2.namedWindow('Aligned RGB-D Frames', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Aligned RGB-D Frames', images)
        key = cv2.waitKey(1)
        
     
        if keyboard.is_pressed('Enter'):
            count += 1
            cv2.imwrite(os.path.join(path_rgb, str(count).zfill(4)+'.png'), color_image)
            cv2.imwrite(os.path.join(path_depth, str(count).zfill(4)+'.png'), depth_image)
            print("Total images captured:", count)
            
            #if count==1:
                #save=depth_image.copy()
                #print("depth_image", save)
                
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            #cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
    
    """
    img_depth=Image(depth_image)
    img_color=Image(color_image)
    rgbd=create_rgbd_image_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)
    """
    img_depth = o3d.io.read_image("o3dtest/Depth/0001.png")
    img_color = o3d.io.read_image("o3dtest/RGB/0001.png")
    
    #check=np.asarray(img_depth)
    #rint("check",check)
    #dif=save-check
    #print(dif)
    
    #depth_scale in *1000[mm]
    #Depth values larger than depth_trunc gets truncated to 0. The depth values will first be scaled and then truncated.
    #(working distance is 1.5m  or  1500mm)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        img_color, img_depth, depth_scale*1000, depth_trunc=2000, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic_od3)
    
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    """ print("type  pcd", type(pcd))
    o3d.visualization.draw_geometries([pcd])"""

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    o3d.visualization.draw_geometries([pcd])
    #o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 2)
    vis.run()
     
    
    
    
    #o3d.visualization.draw_geometries([pcd])
    
    print(type(rgbd_image))
    print(rgbd_image)
    plt.subplot(1, 2, 1)
    plt.title('RGB image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('RGB depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()

