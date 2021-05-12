#Test offline registration first

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


def stereoRGB_depth(square):

    #Inputs:

    #corners [[Nx4] and their corresponding pixels]
    #N is number of markers
    #Recall that order of corners are clockwise
    """Averaging"""
    #Get average x and y coordinate for all Nx4 corners
    print('corners', square)
    #Ensure integer pixels
    
    x_center=[corner[0] for corner in square]
    x_center=sum(x_center)/4
 
    y_center = [corner[1] for corner in square]
    y_center = sum(y_center)/4
    
    z_center=[corner[2] for corner in square]
    z_center=sum(z_center)/4000 # Divide by 4 and 1000 to get in meters
    
    center=[x_center,y_center,z_center]

    center=np.asarray(center)
    
    return center


#####Visualization
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
                                    # zoom=0.4559,                                    #   front=[0.6452, -0.3036, -0.7011],
                                    #   lookat=[1.9892, 2.0208, 1.8945],
                                    #   up=[-0.2779, -0.9482, 0.1556])

#####Extract geometric feature
def preprocess_point_cloud(pcd, voxel_size):
    # We downsample the point cloud, estimate normals, then compute a FPFH feature for each point. 
    # The FPFH feature is a 33-dimensional vector that describes the local geometric property of a point. 
    # A nearest neighbor query in the 33-dimensinal space can return points with similar local geometric structures.
    # See[Rasu2009] for details.
    

    print(":: Downsample with a voxel size %.5f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.5f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size,source,target):
    # Takes a source point cloud and a target point cloud from two files.
    # They are misaligned with an identity matrix as transformation.
    
    print(":: Load two point clouds and disturb initial pose.")
    # source = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_0.pcd")
    # target = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_1.pcd")
    
    #Seems to be a random transformation of the source just for the visualizer 
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    #source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    source_down=source
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_down, target_down, source_fpfh, target_fpfh


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

cad_ref=







"""Load Data"""

#Read source and target PC's
target = o3d.io.read_point_cloud(
    r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\SpineModelKR_V9BacktoMeters_PLY.PLY')
source = o3d.io.read_point_cloud(
    r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\CaptureFrame_PLY32.ply')
"""
voxel_size = 0.00005 # 0.05means 5cm for this dataset
source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
    voxel_size, source, target)
"""

# draw_registration_result(source, target, transformation=np.identity(4))

#o3d.visualization.draw_geometries([source])




"""Global Registration Coarse"""



"""Local Registration (Local)"""


