# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/sensors/realsense_pcd_visualizer.py

# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python

# Based off: https://github.com/intel-isl/Open3D/blob/master/examples/python/reconstruction_system/sensors/realsense_pcd_visualizer.py

"""sources"""

#estimate normals:
#https: // github.com/intel-isl/Open3D/blob/v0.8.0/examples/Python/Basic/pointcloud.py




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

#(Not using preprocessing now)
def preprocess_point_cloud(pointcloud, voxel_size):
    print(":: Downsample with a voxel size %.6f." % voxel_size)
    pointcloud_down = pointcloud.voxel_down_sample(voxel_size=0.05)
    #pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.6f." % radius_normal)
    #estimate_normals(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pointcloud_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pointcloud_down

#Input CAD position coordinates for all aruco position corners and return their center coordinates
#in METERS
def stereoRGB_depth(square):

    #Inputs:

    #corners [[Nx4] and their corresponding pixels]
    #N is number of markers
    #Recall that order of corners are clockwise
    """Averaging"""
    #Get average x and y coordinate for all Nx4 corners
    #print('corners', square)
    #Ensure integer pixels
    
    x_center=[corner[0] for corner in square]
    x_center=sum(x_center)/4000
 
    y_center = [corner[1] for corner in square]
    y_center = sum(y_center)/4000
    
    z_center=[corner[2] for corner in square]
    z_center=sum(z_center)/4000 # Divide by 4 and 1000 to get in meters
    
    center=[x_center,y_center,z_center]

    center=np.asarray(center)
    
    return center


if __name__ == "__main__":
    
    dir = path = r'C: \Users\karla\OneDrive\Documents\GitHub\KUL_Thesis'
    
    """CAD Center Coordinates"""
    #Get Theoretical ARUCO center positions in CAD (ground truth) Coordinates in meters
    #in mm
    
    upperL = [[-105.50, 124.64, - 120.13], [-84.50, 124.64, -120.13],
            [-84.50, 124.63, -99.13], [-105.50, 124.63, -99.13]]
    upperR = [[84.50, 124.64, -120.13], [105.50, 124.64, -120.13],
            [105.50, 124.63, -99.13], [84.50, 124.63, -99.13]]
    lowerL = [[-106.47, 119.50, -28.0], [-85.47, 119.50, -28.00],
            [-85.47, 119.50, -7.00], [-106.47, 119.50, -7.00]]
    lowerR = [[85.47, 119.50, -28.00], [106.47, 119.50, -28.00],
            [106.47, 119.50, -7.00], [85.47, 119.50, -7.00]]

    sup_left = stereoRGB_depth(upperL)
    sup_right = stereoRGB_depth(upperR)
    inf_left = stereoRGB_depth(lowerL)
    inf_right = stereoRGB_depth(lowerR)

    #CAD reference points ( center of aruco markers) in METERS
    #Shape (4,3)
    cad_ref = np.asarray([sup_left, sup_right, inf_right, inf_left])
    #print(cad_ref.shape)
    
    
    """Initialize Parameters for down_sampling PC"""
    voxel_size=1e-15
    radius_normal = voxel_size * 2
    
    
    """Load Ground Truth (CAD) PLY """
    ## in METERS
    
    target = o3d.io.read_point_cloud(
        r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\SpineModelKR_V10ACTUALmeters.PLY')
    
    
    ##target_down = preprocess_point_cloud(target, voxel_size)
    ##print(target)
    
    
    """Estimate Normals"""
    #No downsampling function
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    
    
    """Initialize Camera Parameters and Settings"""
    #Camera Parameter Path
    path = r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\calibration.txt'

    #Load Intrinsics
    param = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = param.getNode("K").mat()
    dist_coef = param.getNode("D").mat()

    #Calll aruco dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
    
    #Aruco marker length
    markerlen = 2.0/100  # Square dimension [m]

    #Initialize aruco IDs used for polygon shape: top left bottom left, top right and  bottom right
    #arucoIDs=[2,35,100,200]
    #21040 KR: add aruco ID's for Male Manikin test and back phantom!
    arucoIDs = [4, 3, 300, 400]
    
    #Initialize Polygon Corners
    poly_corners = [None]*4
    
    #Create empty lists to store marker translation and rotation vectors
    id_rvec=[None]*4
    id_tvec=[None]*4
    
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

    # We will not display the background of objects more than clipping_distance_in_meters meters away
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
                                id_rvec[0], id_tvec[0], markerPoints = aruco.estimatePoseSingleMarkers(
                                    corners[i], markerlen, camera_matrix, dist_coef)
                            elif id == arucoIDs[1]:
                                poly_corners[1] = corn_sq[i, 1, :]
                                id_rvec[1], id_tvec[1], markerPoints = aruco.estimatePoseSingleMarkers(
                                    corners[i], markerlen, camera_matrix, dist_coef)
                            elif id == arucoIDs[2]:
                                poly_corners[2] = corn_sq[i, 2, :]
                                id_rvec[2], id_tvec[2], markerPoints = aruco.estimatePoseSingleMarkers(
                                    corners[i], markerlen, camera_matrix, dist_coef)
                            elif id == arucoIDs[3]:
                                poly_corners[3] = corn_sq[i, 3, :]
                                id_rvec[3], id_tvec[3], markerPoints = aruco.estimatePoseSingleMarkers(
                                    corners[i], markerlen, camera_matrix, dist_coef)
                            else:
                                print('Another ID was detected', id)
                        #Draw the polyline border
                        pts = np.array([poly_corners],np.int32)
                        
                        id_tvec=np.asarray(id_tvec)
                        id_tvec=np.reshape(id_tvec,(4, 3))
                        
                        #Print id_tvecs
                        print('id_tvecs',id_tvec)
                        #print('id_tvecs shape',id_tvec.shape)
                        
                        #Pre-registration transformation matrix
                        pre_reg = initialAlignment(id_tvec, cad_ref)
                        #print(pre_reg)
                        
                        
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
                        
                        """Proceed with PC Visualization"""
                        depth_od3 = o3d.geometry.Image(depth_seg)
                        color_temp_od3 = o3d.geometry.Image(color_seg)
                        
                        """RGBD Generation"""
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
                        
                        #print("Number of points", (np.asarray(temp.points).shape))
                        #Assign values
                        pcd.points = temp.points
                        pcd.colors = temp.colors
                        
                        source=temp
                        
                        """ICP Registration"""
                        # ###source_down = preprocess_point_cloud(source, voxel_size=1e-5)
                        # source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                        
                        # #####TRY ICP#####
                        
                        # #Threshold= the maximum distance in which the search tried to find a correspondence for each point
                        # #Fitness=measures the overlapping area (# of inlier correspondences / # of points in target). The higher the better.
                        # threshold = 10

                        # #Random intial transformation
                        # current_transformation = np.identity(4)
                        
                        # print("source",source.points)
                        # print('temp',target.points)
                        
                        # # reg_p2p = o3d.pipelines.registration.registration_icp(
                        # #     source, target, threshold, current_transformation,
                        # #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
                        
                        
                        # #TransformationEstimation PointToPoint: provides function to compute the residuals and Jacobian matrices of the P2p ICP objective.
                        # #registration_icp takes it as a parameterand runs p2p ICP to obtain results
                        # reg_p2p = o3d.pipelines.registration.registration_icp(
                        #     source, target, threshold, current_transformation,
                        #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20))
                        # print(reg_p2p)
                        # print("Transformation is:")
                        # print(reg_p2p.transformation)
                        
                       
                        
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
                        
                        if keyboard.is_pressed('Enter'):
                            print("Captured")
                            #Write PCD
                            #o3d.io.write_point_cloud("CaptureFrame_PCD"+str(frame_count)+".pcd",temp)
                            #WritePLY
                            o3d.io.write_point_cloud("CaptureBackFrame_PLY"+str(frame_count)+".ply", temp)
                            #save pre-reg as numpy array
                            np.save("preT"+str(frame_count), pre_reg)
                            
                        if keyboard.is_pressed('q'):  # if key 'q' is pressed
                            print('You Pressed quit!')
                            break  # finishing the loop
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
