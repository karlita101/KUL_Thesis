#Purpose: Extract Camera (RGB)  instrinsics for camera calibration

###############################################
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
##      Open CV and Numpy integration        ##
#Source:  https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
###############################################

# the following code was developed using the librealsense GITHUB opencv_viewer_example.py 

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


# Start streaming
pipe_profile=pipeline.start(config)

##(KR) Create a profile to get RGB intrinsics
## video_stream_profile: Stream profile instance which contains additional video attributes.

# Resources:
# Intrinsic Parameters-  Resource[1] 
# ('https://bit.ly/3nQTprF')
#Intrinsic Parameters- Main one to develop code[2] 
#('https://bit.ly/3j034bX')
# Camera Calibration [3]
#'https://nikatsanka.github.io/camera-calibration-using-opencv-and-python.html'

try:
    while  True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue # https://www.tutorialspoint.com/python/python_loop_control.htm {info on continue}
        #].profile' Stream profile from frame handle. Identical to calling get_profile.
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile) # get_extrinsics_to(...)	Get the extrinsic transformation between two profiles (representing physical sensors)
        depth_sensor = pipe_profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)
        #depth_pixel = [200, 200]
        #depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_scale)
        #color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
        #color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)
        
        print('Color Camera Intrinsics are',color_intrin)
        print(color_intrin.fx)
         # Press esc or 'q' to close the image window
        if KeyboardInterrupt:
            break
finally:  
    #Stope pipeline 
    pipeline.stop()

