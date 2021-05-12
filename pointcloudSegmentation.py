# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/sensors/realsense_pcd_visualizer.py

# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python

# Based off: https://github.com/intel-isl/Open3D/blob/master/examples/python/reconstruction_system/sensors/realsense_pcd_visualizer.py
###################################################################################


import pyrealsense2 as rs
import numpy as np
from enum import IntEnum

from datetime import datetime
import open3d as o3d
import cv2
import cv2.aruco as aruco
import os
import keyboard



class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = o3d.camera.PinholeCameraIntrinsic(640, 480, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)
    return out


if __name__ == "__main__":

    #Camera Parameter Path
    path = r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\calibration.txt'

    #Load Intrinsics
    param = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = param.getNode("K").mat()
    dist_coef = param.getNode("D").mat()

    #Calll aruco dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)

    #Initialize aruco IDs used for polygon shape: top left bottom left, top right and  bottom right
    #arucoIDs=[2,35,100,200]
    #21040 KR: add aruco ID's for Male Manikin test
    arucoIDs = [4, 3, 300, 400]
    
    #Initialize Polygon Corners
    poly_corners = [None]*4
    
    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    #KR Turn ON Emitter
    depth_sensor.set_option(rs.option.emitter_always_on, 1)

    # Get depth sensor's depth scale
    depth_scale = depth_sensor.get_depth_scale()

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3  # 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames. Here RGB.
    align_to = rs.stream.color
    align = rs.align(align_to)

    #initialize visualizer Class
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd = o3d.geometry.PointCloud()
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

    # Streaming loop
    frame_count = 0
    #frame_count= False
    try:
        while True:

            dt0 = datetime.now()

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                get_intrinsic_matrix(color_frame))

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            #Numpy array equivalent
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            
            #Detect Aruco
            #Grayscale 
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            arucoParameters = aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                gray_image, aruco_dict, parameters=arucoParameters, cameraMatrix=camera_matrix, distCoeff=dist_coef)

            #Draw detected markers on RGB image
            color_image=aruco.drawDetectedMarkers(color_image, corners,ids)
            aruco.drawDetectedMarkers(color_image, corners)
            corn_sq = np.squeeze(corners)
            
            #Markers were detected
            if ids is not None:
                
                #If 4 Markers were detected
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
                        
                        #Draw Polygon Border
                        cv2.polylines(color_image, np.array([pts]), True, (0,0,255), 5)
                        
                        #Create a binary mask ( 1 channel)
                        binary_mask=np.zeros((gray_image.shape),np.uint8)
                        cv2.fillPoly(binary_mask, [pts], (255, 255, 255),8)

                        #Segment Depth and  RGB frame with binary mask 
                        
                        depth_seg = cv2.bitwise_and(depth_image, depth_image, mask=binary_mask)
                        color_seg=cv2.bitwise_and(color_image, color_image, mask=binary_mask)
                        #cv2.imshow("depth seg",depth_seg)
                        #cv2.waitKey(25)
                        cv2.imshow("color seg",color_seg)
                        #color_selection[binary_mask==0]=255
                        
                        #Convert Color to RGB
                        color_seg = cv2.cvtColor(color_seg, cv2.COLOR_BGR2RGB)
                        ###########Proceed with PC Visualization########
                        depth_od3 = o3d.geometry.Image(depth_seg)
                        color_temp_od3 = o3d.geometry.Image(color_seg)
                        
                        #RGBD Generation
                        #Open3D assumed that color and depth image are synchronized and registered in the same coordinate frame
                        #Set to false to preserve 8-bit color channels
                        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                            color_temp_od3,
                            depth_od3,
                            depth_scale=1.0 / depth_scale,
                            depth_trunc=clipping_distance_in_meters,
                            convert_rgb_to_intensity=False)
                        
                        temp = o3d.geometry.PointCloud.create_from_rgbd_image(
                            rgbd_image, intrinsic)
                        temp.transform(flip_transform)
                        pcd.points = temp.points
                        pcd.colors = temp.colors
                        
                        if frame_count == 0:
                            #if frame_count== False:
                            vis.add_geometry(pcd)
                        
                        #Update_geometry
                        vis.update_geometry(pcd)
                        #Render new frame
                        vis.poll_events()
                        vis.update_renderer()
                        
                        process_time = datetime.now() - dt0
                        print("FPS: " + str(1 / process_time.total_seconds()))
                        frame_count += 1
                        #frame_count=True
                        
                else:
                    pts = None
                    print("Non-intialized Aruco detected")
            else:
                pts=None
                print("No Aruco markers detected")



            
            
            ######################################
            # Render images:
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))

            cv2.namedWindow('Aligned RGB-D Frames', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Aligned RGB-D Frames', images)
            key = cv2.waitKey(1)
            ######################################
            
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        pipeline.stop()
    #Close Open3D Window
    vis.destroy_window()
