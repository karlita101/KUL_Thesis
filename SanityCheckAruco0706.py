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

import matplotlib.pyplot as plt

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
    

    # """Initialize Parameters for down_sampling PC"""
    # voxel_size=1e-15
    # radius_normal = voxel_size * 2
    
    
    """Load Ground Truth (CAD) PLY """
    ## in METERS
    # source = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\SpineModelKR_V12UpperSurface.PLY')
    
    # #assembly = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\SpineModelKR_rev2_fine.PLY')
    # assembly = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\SpineBack_Assembly_KR_rev2.PLY')
    
      
    # voxel_size = 0.10/100  # 1cm
    # down_assembly = assembly.voxel_down_sample(voxel_size)
    # down_assembly.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=30))
    
   
    # """"Create Deep Copies"""
    # #Source Copy (Orange)
    # source_temp = copy.deepcopy(source).paint_uniform_color([1, 0.706, 0])
    # assembly_temp = copy.deepcopy(down_assembly).paint_uniform_color([0.44, 0.53, 0.6])

    
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
    
    #read 50 cases
    n_read=100
    
    #x_vec = np.array([], dtype=np.int64).reshape(0, 4)
    x_vec=np.zeros((n_read,4))
    count_read=[]
    
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

    # #initialize visualizer Class
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # pcd = o3d.geometry.PointCloud()
    # flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

    # #June 28: path
    # path_pc = o3d.geometry.PointCloud()
    
    # Streaming loop
    frame_count = 0

    read=0
    
    #frame_count= False
    try:
        # start_dt=time.time()
        # end_dt=start_dt+60*0.5
        
        while True:
        #while time.time()<end_dt:

            dt0 = datetime.now()

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            # intrinsic = o3d.camera.PinholeCameraIntrinsic(get_intrinsic_matrix(color_frame))

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
                        
                        if read <n_read:
                            id_tvec_trans=np.transpose(id_tvec)
                            print("id_tvec",id_tvec)
                            print("------transpose ---")
                            print(id_tvec_trans)
                            print("first row")
                            print(id_tvec_trans[0, :])
                        
                            xrow = np.reshape(id_tvec_trans[0, :], (1, 4))
                            
                            x_vec[read,:]=xrow
                            #np.vstack((x_vec,xrow))
                            #np.append(x_vec,id_tvec_trans[0,:],axis=0)
                            #delta_t=time.time()-start_dt
                            #np.append(count_read,read)
                            count_read.append(read)
                            read += 1
                        
                            
                        #Print id_tvecs
                        #print('id_tvecs',id_tvec)
                        #print('id_tvecs shape',id_tvec.shape)
                        

                        """difb_01=id_tvec[0]-id_tvec[1]
                        difb_12 = id_tvec[1]-id_tvec[2]
                        ##Get distances!
                        normb_01=np.linalg.norm(difb_01)
                        # print("norm CAD from 0 to 1",norma_01)
                        # print("norm RS from 0 to 1", normb_01)
                        normb_12=np.linalg.norm(difb_12)
                        #Assign to array
                        norm_ARUCO=np.array([normb_01,normb_12])
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
                        
                        # """Proceed with PC Visualization"""
                        # depth_od3 = o3d.geometry.Image(depth_seg)
                        # color_temp_od3 = o3d.geometry.Image(color_seg)
                        
                        # """Create RGBD"""
                        # #Open3D assumed that color and depth image are synchronized and registered in the same coordinate frame
                        # #Set to false to preserve 8-bit color channels
                        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        #     color_temp_od3,
                        #     depth_od3,
                        #     depth_scale=1.0 / depth_scale,
                        #     depth_trunc=clipping_distance_in_meters,
                        #     convert_rgb_to_intensity=False)
                        
                        # temp = o3d.geometry.PointCloud.create_from_rgbd_image(
                        #     rgbd_image, intrinsic)
                        # temp.transform(flip_transform)
                        
                        # #print("Number of points", (np.asarray(temp.points).shape))
                        # #Assign values
                        # pcd.points = temp.points
                        # pcd.colors = temp.colors
                        
                    
                        # """ICP Registration"""
                        # #Use the preregistration to speed up ICP
                        # T = pre_reg
                        # T=np.matmul(flip_transform,T)


                        # threshold = 80/100  # [m] 	Maximum correspondence points-pair distance
                        # reg_p2p = o3d.pipelines.registration.registration_icp(
                        #     source_temp, pcd, threshold, T,
                        #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
                        # print("ICP evalutation",reg_p2p)
                        # print("Transformation is:")
                        # print(reg_p2p.transformation)

                        # """Get Registered  PC"""
                        
                        # #assembly_temp.transform(reg_p2p.transformation)
                        
                        # #Sking (Back) ONLY- WOKRS
                        # source_icp =source_temp.transform(reg_p2p.transformation).paint_uniform_color([0, 0.651, 0.929])

                        # assembly_icp=assembly_temp.transform(reg_p2p.transformation).paint_uniform_color([1,0, 0])
                        
                        
                        
                        # #Generate Grid Path
                        # """p2=id_tvec[0]
                        # p3=id_tvec[2]
                        # p1=id_tvec[3]
                       
                        # w=2
                        # l=3
                        
                        # path=[]
                        # for i in range(l+1):
                        #     for j in range(w+1):
                        #         #print("i",i,"j",j)
                        #         path.append(p1+i/l*(p3-p1)+j/w*(p2-p1))
                        
                        # # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
                        # #path_pc = o3d.geometry.PointCloud()
                        # path_pc.points = o3d.utility.Vector3dVector(path)
                        # #Color
                        # path_pc.paint_uniform_color([1.0, 0.0, 1.0])
                        # #calculate Normals
                        # #enter 'n' in keyboard to see normals
                        # path_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
                        # path_pc.transform(flip_transform)
                        
                        # start_point = 4
                        # end_point = 7

                        # trajectory = gettrajectory(start_point, end_point, w, l, path)"""


                        
                        # #"""Visualize"""
                       
                        # if frame_count == 0:
                        #     vis.add_geometry(pcd)
                        #     vis.add_geometry(assembly_icp)
                        #     ##vis.add_geometry(source_icp)
                        #     #vis.add_geometry(path_pc)
                        
                        
                        # #Update_geometry
                        # vis.update_geometry(pcd)
                        # vis.update_geometry(assembly_icp)
                        # #vis.update_geometry(source_temp)
                        # vis.update_geometry(source_icp)
                        # #vis.update_geometry(path_pc)
                        
                        # #Render new frame
                        # vis.poll_events()
                        # vis.update_renderer()
                        
                        # process_time = datetime.now() - dt0
                        # print("FPS: " + str(1 / process_time.total_seconds()))
                        # frame_count += 1
                        # #frame_count=True
                        
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
    # vis.destroy_window()
    
    #Calculate key distance from CAD model ( ground truth)
    #0-1: Superior Left to Superior Right
    #1-2: Superior Right to Inferior Right
    difa_01 = cad_ref[0]-cad_ref[1]
    difa_12 = cad_ref[1]-cad_ref[2]
    norma_01 = np.linalg.norm(difa_01)
    norma_12 = np.linalg.norm(difa_12)

    print("norm cad 01",norma_01)
    print("norm cad 12", norma_12)
    
    
    print('------x_vec----')
    print(x_vec)
    
    t=np.array(range(n_read))
    # fig1 = plt.figure()
    # plt.plot(t, x_vec[:, 0])
    # plt.plot(t, x_vec[:, 1])
    # plt.plot(t, x_vec[:, 2])
    # plt.plot(t, x_vec[:, 3])
    # plt.ylabel('some numbers')
    # fig1.savefig("test.png")
    # plt.show() 
    # plt.pause(5)
    # #fig.show()
    # print("stop")



    fig2, ax2 = plt.subplots()
    l0,=ax2.plot(t,  x_vec[:, 0]*1000)
    l1,=ax2.plot(t,  x_vec[:, 1]*1000)
    l2,=ax2.plot(t,  x_vec[:, 2]*1000)
    l3,=ax2.plot(t,  x_vec[:, 3]*1000)
    
    ax2.set(xlabel='samples', ylabel='X Position (mm)',title='Detected ArUco Marker X Postition')
    ax2.legend((l0, l1, l2, l3), (str(arucoIDs[0]), str(arucoIDs[1]), str(arucoIDs[2]), str(arucoIDs[0])),
               loc='upper right', shadow=True)
    
    #ax2.grid()
    #plt.ticklabel_format(axis='both', style='sci', scilimits=(3,3))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    fig2.savefig("test2.png")
    plt.show()
    
    figx, axs = plt.subplots(2, 2)

    # basic plot
    axs[0, 0].boxplot(x_vec[:, 0]*1000)
    axs[0, 0].set_title('basic plot')

    axs[0, 1].boxplot(x_vec[:, 1]*1000)
    axs[0, 1].set_title('basic plot')
    
    axs[1, 0].boxplot(x_vec[:, 3]*1000)
    axs[1, 0].set_title('basic plot')
    
    axs[1, 1].boxplot(x_vec[:, 2]*1000)
    axs[1, 1].set_title('basic plot')
    
    figx.savefig("testx.png")
    
    
    #data_to_plot = np.random.rand(100, 5)
    # positions = np.arange(4) + 1

    # fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    # # matplotlib > 1.4
    # bp = ax[0].boxplot(x_vec, positions=positions, showmeans=True)
    # ax[0].set_title("Using showmeans")
    # fig.savefig("test3.png")
    
    
