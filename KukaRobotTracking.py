import pyrealsense2 as rs
import numpy as np
from enum import IntEnum

from datetime import datetime
import open3d as o3d
import cv2
import cv2.aruco as aruco
import os
import keyboard

from preregistration import *

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def get_intrinsic_matrix(frame):
    #Get from intelrealsensecamera directly
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = o3d.camera.PinholeCameraIntrinsic(640, 480, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)
    return out


if __name__ == "__main__":
    
    dir = path = r'C: \Users\karla\OneDrive\Documents\GitHub\KUL_Thesis'
     
    """Initialize Camera Parameters and Settings"""
    #Camera Parameter Path
    path = r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\calibration.txt'

    #Load Intrinsics
    param = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = param.getNode("K").mat()
    dist_coef = param.getNode("D").mat()

    """ARUCO"""
    #Calll aruco dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
    
    #marker length
    #3.19
    markerlen = 2/100
    #Drawn axis length
    axis_len = 0.02

    #Create empty lists to store marker translation and rotation vectors
    id_rvec=[]
    id_tvec=[]
    
    #Tracking Aruco Markers
    track_rvec=[]
    track_tvec=[]
    track_id=[]
    # Create a pipeline
    pipeline = rs.pipeline()

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

    # We will not display the background of objects more than clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3  # 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames. Here RGB.
    align_to = rs.stream.color
    align = rs.align(align_to)


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
            
            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            #Numpy array equivalent
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            
            """Detect Aruco"""
            #Grayscale 
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            arucoParameters = aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                gray_image, aruco_dict, parameters=arucoParameters, cameraMatrix=camera_matrix, distCoeff=dist_coef)

            print("ids",ids)
            """Draw detected markers on RGB image"""
            color_image=aruco.drawDetectedMarkers(color_image, corners,ids)
            aruco.drawDetectedMarkers(color_image, corners)
            corn_sq = np.squeeze(corners)
            
            #Markers were detected
            if ids is not None:                             
                for i, id in enumerate(ids):
                    #Get translation and rotation vectors (1,1,3 shape)
                    print(i)
                    print('Corresponding ID value is', id)
                    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], markerlen, camera_matrix, dist_coef)
                    #Store Rvec and Tvec  values
                    id_rvec.append(rvec)
                    id_tvec.append(tvec)

                    #Draw aruco pose axes
                    aruco.drawAxis(color_image, camera_matrix,dist_coef, rvec, tvec, axis_len)

                process_time = datetime.now() - dt0
                print("FPS: " + str(1 / process_time.total_seconds()))
                frame_count += 1
                #frame_count=True     
                                                  
                #Save Markers t, r vectors 
                if keyboard.is_pressed('Enter'):                           
                    np.save('./210628KukaRegistration/'+'arucoTvec'+str(frame_count), track_tvec)
                    np.save('./210628KukaRegistration/'+'arucoRVec='+str(frame_count), track_rvec)
                    np.save('./210628KukaRegistration/'+'idOrder='+str(frame_count), ids)
                    print("Captured")
                    
                if keyboard.is_pressed('q'):  # if key 'q' is pressed
                    print('You Pressed quit!')
                    break  # finishing the loop
               
                   
            else:
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


