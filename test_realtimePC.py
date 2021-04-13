import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import os
import keyboard
from imutils import perspective
from datetime import datetime
from open3d import *


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



align_to = rs.stream.color
align = rs.align(align_to)


depth_sensor = profile.get_device().first_depth_sensor()


# Streaming loop
try:
    vis = open3d.visualization.Visualizer()
    vis.create_window("Tests")
    pcd = open3d.geometry.PointCloud()
    
    while True:
        dt0 = datetime.now()
        vis.add_geometry(pcd)
        pcd.clear()
        
        # Get frameset of color and depth

        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        

        if not aligned_depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        
        pc = rs.pointcloud()

        pc.map_to(color_frame)
        points = pc.calculate(aligned_depth_frame)
        print(type(points))
        vtx = np.asanyarray(points.get_vertices())
        print(type(vtx))
        print(vtx.shape)
        
        pcd.points = open3d.utility.Vector3dVector(vtx)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        process_time = datetime.now() - dt0
        print("FPS = {0}".format(1/process_time.total_seconds()))

finally:
    pipeline.stop()
