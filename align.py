## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

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
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to) # K:  alignn color to depth



#K
#align depth frames to their corresponding color frames. We generate a new frame sized as color
#  stream but the content being depth data calculated in the color sensor coordinate system.
#  In other word to reconstruct a depth image being "captured" using the origin and dimensions of the color sensor.
#Then, we use the original color and the re-projected depth frames (which are aligned at this stage) to determine the
 #depth value of each color pixel.

#A rs2::align object transforms between two input images, from a source image to some target image which is specified
#with the align_to parameter.







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

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 50
        alt_color = (255, 0, 0)
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        #print("depth image 3D: ", depth_image_3d)
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        #qqprint("bg_removed: ", bg_removed)

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #KarlaL here we are aplying it to a 1x wtahever array. Scale relativeto the depth infor==> get color mapping. color mapping 3 values!
        images = np.hstack((bg_removed, depth_colormap))
        #print("images: ", images)
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()