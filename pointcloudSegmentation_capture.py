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
import copy
import time
import math
from preregistration import *
import matplotlib.pyplot as plt

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

#Trajectory path in vertical lienes bottom>  up
def gettrajectory(start_point, end_point, w, l, path):
    scan_start = start_point//(w+1)
    scan_end = end_point//(w+1)

    scan = scan_start
    initial_point = start_point

    trajectory = []

    while (scan <= scan_end) and (scan >= scan_start):
        #print("initial_point", initial_point)
        #print("scan", scan)

        #End value to scan to for each scan line
        val_end = (scan+1)*(w+1)
        #print('val end', val_end)
        #if end value is within the same scan line
        if (scan < (val_end)//(w+1)) and (end_point < val_end):
            traj = path[initial_point:end_point+1]
            #print(traj)
            #print("Partial line")
        #complete the whole scan line, and move on
        else:
            traj = path[initial_point:val_end]
            #print(traj)
            #print("Whole line")
        #add grid path to global trajectory
        np.append(trajectory, traj)
        #update intial point to start off on the next interation
        initial_point = (scan+1)*(w+1)
        #next scan line
        scan += 1
        
def getmattransform(Rmat,tvec):
    #rodrigues to rot matrix
    T_mat=np.zeros((4,4))
    T_mat[:3,:3]=Rmat
    #tvec = np.squeeze(tool_tvec).reshape(3, 1)
    T_mat[:3,3]=tvec
    T_mat[3,3]=1
    return T_mat

# Calculates Rotation Matrix given euler angles.


def eulerAnglesToRotationMatrix(theta):
    #XYZ
    R_x = np.array([[1,         0,                  0],
                    [0,         math.cos(theta[0]), -math.sin(theta[0])],
                    [0,         math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])],
                    [0,                     1,      0],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    Rxyz = np.dot(R_x, np.dot(R_y,R_z))
    return Rxyz

    
    
if __name__ == "__main__":
    
    dir = path = r'C: \Users\karla\OneDrive\Documents\GitHub\KUL_Thesis'
    
    """CAD Center Coordinates"""
    #CAD aruco Markers vertices coordinate [x,y,z] in mm    
    
    upperL = [[-105.50, 124.64, - 120.13], [-84.50, 124.64, -120.13],
            [-84.50, 124.63, -99.13], [-105.50, 124.63, -99.13]]
    upperR = [[84.50, 124.64, -120.13], [105.50, 124.64, -120.13],
            [105.50, 124.63, -99.13], [84.50, 124.63, -99.13]]
    lowerL = [[-106.47, 119.50, -28.0], [-85.47, 119.50, -28.00],
            [-85.47, 119.50, -7.00], [-106.47, 119.50, -7.00]]
    lowerR = [[85.47, 119.50, -28.00], [106.47, 119.50, -28.00],
            [106.47, 119.50, -7.00], [85.47, 119.50, -7.00]]

    upperL = np.asarray(upperL)
    cen_UL = np.mean(upperL, axis=0)

    upperR = np.asarray(upperR)
    cen_UR = np.mean(upperR, axis=0)

    lowerL = np.asarray(lowerL)
    cen_LL = np.mean(lowerL, axis=0)

    lowerR = np.asarray(lowerR)
    cen_LR = np.mean(lowerR, axis=0)

    # (4,3) shapate in [m] units
    cad_ref = np.asarray([cen_UL, cen_UR, cen_LR, cen_LL])/1000
    
    #Get dimension lengths and width (TRUE=CAD)
    difa_01 = cad_ref[0]-cad_ref[1]
    difa_12 = cad_ref[1]-cad_ref[2]

    difa_32 = cad_ref[3]-cad_ref[2]
    difa_03 = cad_ref[0]-cad_ref[3]

    norma_01 = np.linalg.norm(difa_01)
    norma_12 = np.linalg.norm(difa_12)

    norma_32 = np.linalg.norm(difa_32)
    norma_03 = np.linalg.norm(difa_03)

    length_true = np.array([norma_01, norma_32])*1000
    width_true = np.array([norma_12, norma_03])*1000

    # print("------true length in mm------")
    # print(length_true)
    # print("------true width in mm------")
    # print(width_true)
    
   
    

    # """Initialize Parameters for down_sampling PC"""
    # voxel_size=1e-15
    # radius_normal = voxel_size * 2
    
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    
    """Load Ground Truth (CAD) PLY """
    ## in METERS
    source = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\SpineModelKR_V12UpperSurface.PLY')
    
    #assembly = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\SpineModelKR_rev2_fine.PLY')
    assembly = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\SpineBack_Assembly_KR_rev2.PLY')
          
    voxel_size = 0.10/100  # 1cm
    down_assembly = assembly.voxel_down_sample(voxel_size)
    down_assembly.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=30))

    #tool
    #tool = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\0708MockInstrument_rev1_PLY.PLY')
    tool_mesh = o3d.io.read_triangle_mesh(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\0708MockInstrument_rev1_PLY.PLY')
    #tool_mesh.transform(flip_transform)
    toolID= 84
    
    
    """"Create Deep Copies"""
    #Source Copy (Orange)
    source_temp = copy.deepcopy(source).paint_uniform_color([1, 0.706, 0])
    assembly_temp = copy.deepcopy(down_assembly).paint_uniform_color([0.44, 0.53, 0.6])
    
    
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
    print("Depth Scale is: ", depth_scale)

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
    

    #June 28: path
    path_pc = o3d.geometry.PointCloud()
    
    # Streaming loop
    frame_count = 0

    #>= 4 Aruco markers read count
    read=0
    
    
    #Plot in real time
    fig = plt.figure()
    axe = fig.add_subplot(111)
    axe.set_ylabel('Difference in mm')
    axe.set_title('Measured difference in AruCo and CAD dimension')
    X, Y = [], []
    sp, = axe.plot([], [], label='toto', ms=10, color='k', marker='o', ls='')
    fig.show()
    
    
    """initalize point cloud objects""" 
    source_pcd=o3d.geometry.PointCloud()
    assemb_pcd=o3d.geometry.PointCloud()
    
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

            
            """Detect Aruco"""
            #Grayscale 
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            arucoParameters = aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                gray_image, aruco_dict, parameters=arucoParameters, cameraMatrix=camera_matrix, distCoeff=dist_coef)

            """Draw detected markers on RGB image"""
            color_image=aruco.drawDetectedMarkers(color_image, corners,ids)
            aruco.drawDetectedMarkers(color_image, corners)
            corn_sq = np.squeeze(corners)
            
            #Markers were detected
            if ids is not None:
                
                #Tool Flag: False= no tool exists
                tool_flag = False
                
                #If 4 Markers were detected
                if len(corners) >= 4:
                    
                        for i, id in enumerate(ids):
                            #print('ID index', i, 'Value',id)
                            if id == arucoIDs[0]:
                                poly_corners[0] = corn_sq[i, 0, :]
                                id_rvec[0], id_tvec[0], markerPoints = aruco.estimatePoseSingleMarkers(
                                    corners[i], markerlen, camera_matrix, dist_coef)
                                # #Get center pixel coordinates
                                print("------SHAPE OF ALL CORNER--------")
                                center = corn_sq[i, :, :]
                                # print(center)
                                # print(center.shape)
                                # print("--------Mean of pixel coordinates to get center")
                                # print(np.mean(center, axis=0).astype(int))
                                
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
                            elif id == toolID:
                                
                                #Flag that the tool exist!
                                tool_flag=True
                                
                                #Estimate Tool Pose from Aruco 
                                tool_rvec, tool_tvec, markerPoints = aruco.estimatePoseSingleMarkers(
                                    corners[i], markerlen, camera_matrix, dist_coef)
                                
                                # print("----TOOL INFO-----")
                                # print(tool_rvec)
                                # print(tool_tvec)
                                # print(np.squeeze(tool_tvec).shape)
                                
                                """Get PC ---> Aruco Rot Matrix"""
                                #Tranformation Matrix from PointCloud to Aruco Coordinate Frame
                                angles = [0, np.pi, np.pi]
                                Rxyz=eulerAnglesToRotationMatrix(angles)
                                
                                """Get Aruco --->PC Transform Matrix"""
                                #Tranpose of Rxyz
                                Rxyz_trans=np.transpose(Rxyz)
                                Txyz = getmattransform(Rxyz_trans, np.zeros(3))
                            
                                
                                """Get Aruco Transformation Matrix"""
                                tool_Rmat, __ = cv2.Rodrigues(tool_rvec)
                                tool_transform=getmattransform(tool_Rmat,tool_tvec)
                                print(tool_transform)
                                
                                """Transform ARUCO Tmax to PC Coordinate Frame """
                                #PC_cad= Taruco2PC * Taruco
                                #PC_cad= Txyz*Taruco
                                tool_aruco2PC = np.matmul(Txyz, tool_transform)
                                
                                """Apply to CAD Model"""
                                
 
                                tool=copy.deepcopy(tool_mesh).transform(tool_aruco2PC)
                            else:
                                print('Another ID was detected', id)
                        
                        #Check if all 4 ARUCO markers to segment back were detected!
                        if poly_corners is not None:
                            
                            #Draw the polyline border
                            pts = np.array([poly_corners],np.int32)
                            
                            id_tvec=np.asarray(id_tvec)
                            id_tvec=np.reshape(id_tvec,(4, 3))
                            
                            #Print id_tvecs
                            #print('-------id_tvecs-------',id_tvec)
                            #print('id_tvecs shape',id_tvec.shape)
                            
                            #Jul 07: Use to check if the RS depth ant Z-tvec corresponse for the center pixel
                            print("----- Marker 1 RS depth value-------")
                            #Right now only marker
                            coord=np.mean(center, axis=0).astype(int)
                            print([coord[1],coord[0]])
                            d = depth_image[coord[1], coord[0]]
                            print('d',d)
                            print(d*depth_scale)
                            print(id_tvec[0,2])
                            print("error in depth in mm", (d*depth_scale-id_tvec[0,2])*1000)
                            
                            #Subtract vectors
                            difb_01 = id_tvec[0]-id_tvec[1]
                            difb_12 = id_tvec[1]-id_tvec[2]
                            difb_32 = id_tvec[3]-id_tvec[2]
                            difb_03 = id_tvec[0]-id_tvec[3]
                            
                            ##Get distances! (NORM)
                            normb_01=np.linalg.norm(difb_01)
                            normb_12=np.linalg.norm(difb_12)
                            
                            normb_32 = np.linalg.norm(difb_32)
                            normb_03 = np.linalg.norm(difb_03)
                            
                            #Arrange in length and width dim
                            length_dim = np.array([normb_01, normb_32])*1000
                            width_dim = np.array([normb_03, normb_12])*1000
                            
                            
                            # print("norm CAD from 0 to 1",norma_01)
                            # print("norm RS from 0 to 1", normb_01)
                            
                            # print("------Length Dimensions in mm--------")
                            # print(length_dim)
                            # print("------Widthth Dimensions in mm--------")
                            # print(width_dim)
                            
                            #Difference from TRUE (CAD)
                            length_dif=length_dim-length_true
                            width_dif=width_dim-width_true
                            
                            print("-----DIF LENGTH-----")
                            print(length_dif)
                            print("-----DIF WIDTH------")
                            print(width_dif)
                            
                            print("-----percent error-----")
                            print(length_dif/length_true*100)
                            print(width_dif/width_true*100)
                            
                            X.append(read)
                            Y.append(length_dif[0])
                            sp.set_data(X, Y)
                            axe.set_xlim(min(X), max(X))
                            axe.set_ylim(min(Y), max(Y))
                            #raw_input('...')
                            fig.canvas.draw()

                            
                            read+=1

                            #Assign to array
                            ##JULY7: norm_ARUCO=np.array([normb_01,normb_12])
                            # print("norm CAD from 1 to 2",norma_12)
                            # print("norm RS from 1 to 2", normb_12)"""
                            
                            """Pre-registration transformation matrix"""
                            #a= CAD ,
                            #b= ARUCO
                            #SOURCE=CAD
                            #TARGET=Intel                                               
                            pre_reg = initialAlignment(cad_ref,id_tvec)
                            #print(pre_reg)
                            
                            """Draw Polygon Border"""
                            cv2.polylines(color_image, np.array([pts]), True, (0,0,255), 5)
                            
                            """Create a binary mask ( 1 channel)"""
                            binary_mask=np.zeros((gray_image.shape),np.uint8)
                            cv2.fillPoly(binary_mask, [pts], (255, 255, 255),8)

                            """Segment Depth and  RGB frame with binary mask """
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
                            
                            """Create RGBD"""
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
                            
                        
                            """ICP Registration"""
                            #Use the preregistration to speed up ICP
                            T = pre_reg
                            T=np.matmul(flip_transform,T)
                            
                            
                            ##July7: always do icp with the deepcopy of source
                            threshold = 80/100  # [m] 	Maximum correspondence points-pair distance
                            reg_p2p = o3d.pipelines.registration.registration_icp(
                                copy.deepcopy(source), pcd, threshold, T,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
                            
                            
                            #July 7: comment out
                            # print("ICP evalutation",reg_p2p)
                            # print("Transformation is:")
                            # print(reg_p2p.transformation)

                            """Get Registered  PC"""
                            # print(np.asarray(source_temp.points)[0])
                            # print(np.asarray(source.points)[0])

                            
                            source_icp=copy.deepcopy(source_temp).transform(reg_p2p.transformation).paint_uniform_color([0, 0.651, 0.929])                            
                            source_pcd.points=source_icp.points
                            source_pcd.colors=source_icp.colors
                            
                            assemb_icp=copy.deepcopy(assembly_temp).transform(reg_p2p.transformation).paint_uniform_color([0, 0.651, 0.929])                            
                            assemb_pcd.points=assemb_icp.points
                            assemb_pcd.colors = assemb_icp.colors
                            
                            
                            #tool=copy.deepcopy(tool_mesh).transform(flip_transform)
                            
                            
                            #Generate Grid Path
                            """p2=id_tvec[0]
                            p3=id_tvec[2]
                            p1=id_tvec[3]
                        
                            w=2
                            l=3
                            
                            path=[]
                            for i in range(l+1):
                                for j in range(w+1):
                                    #print("i",i,"j",j)
                                    path.append(p1+i/l*(p3-p1)+j/w*(p2-p1))
                            
                            # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
                            #path_pc = o3d.geometry.PointCloud()
                            path_pc.points = o3d.utility.Vector3dVector(path)
                            #Color
                            path_pc.paint_uniform_color([1.0, 0.0, 1.0])
                            #calculate Normals
                            #enter 'n' in keyboard to see normals
                            path_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
                            path_pc.transform(flip_transform)
                            
                            start_point = 4
                            end_point = 7

                            trajectory = gettrajectory(start_point, end_point, w, l, path)"""


                            
                            """Visualize"""
                            if frame_count == 0:
                                vis.add_geometry(pcd)
                                vis.add_geometry(assemb_pcd)
                                
                                #try something else for conditional viewing:
                                
                                
                                #vis.add_geometry(source_pcd)
                                #vis.add_geometry(path_pc)
                            
                            #Update_geometry
                            vis.update_geometry(pcd)
                            vis.update_geometry(assemb_pcd)
                                                        
                            if tool_flag == True:
                                vis.add_geometry(tool)
                                vis.update_geometry(tool)                            
                                
                                
                            #vis.update_geometry(source_pcd)
                            #vis.update_geometry(path_pc)
                            
                            #Render new frame
                            vis.poll_events()
                            vis.update_renderer()
                            
                            if tool_flag==True:
                                vis.remove_geometry(tool)
                                
                            process_time = datetime.now() - dt0
                            print("FPS: " + str(1 / process_time.total_seconds()))
                            frame_count += 1
                            #frame_count=True
                        
                            """if keyboard.is_pressed('Enter'):
                            
                            #Write PCD
                            #o3d.io.write_point_cloud("CaptureFrame_PCD"+str(frame_count)+".pcd",temp)
                            #WritePLY
                            
                            
                            #210623PilotTestInvestigatePC
                            #./210517PilotTest/preregmat/'+"preregT"
                            #./210623PilotTestInvestigatePC

                            o3d.io.write_point_cloud(
                                "./210624PilotTestAngles60/Angle30/pointclouds/"+"BackPLY"+str(frame_count)+".ply", temp)
                            #save pre-reg as numpy array
                            np.save(
                                './210624PilotTestAngles60/Angle30/preregmat/'+"preregT"+str(frame_count), pre_reg)
                            
                            #save aruco marker coordinates
                            np.save(
                                './210624PilotTestAngles60/Angle30/arucotvec/'+'id_tvec'+str(frame_count), id_tvec)
                            print("Captured")
                            
                            #save aruco marker distances
                            #np.save(
                            #    './210517PilotTest/distancesnpy/'+'normdist'+str(frame_count), norm_ARUCO)
                            #print("Captured")"""
                            
                            
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
    
    #Calculate key distance from CAD model ( ground truth)
    #0-1: Superior Left to Superior Right
    #1-2: Superior Right to Inferior Right
    """difa_01 = cad_ref[0]-cad_ref[1]
    difa_12 = cad_ref[1]-cad_ref[2]
    
    difa_32 = cad_ref[3]-cad_ref[2]
    difa_03 = cad_ref[0]-cad_ref[3]
    
    norma_01 = np.linalg.norm(difa_01)
    norma_12 = np.linalg.norm(difa_12)
    
    norma_32 = np.linalg.norm(difa_32)
    norma_03 = np.linalg.norm(difa_03)
    
    length_true=np.array([norma_01,norma_32])*1000
    width_true = np.array([norma_12, norma_03])*1000
    
    print("------true length in mm------")
    print(length_true)
    print("------true width in mm------")
    print(width_true)"""
    
    # print("norm cad 01",norma_01)
    # print("norm cad 12", norma_12)
