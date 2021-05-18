#Test offline registration first

import pyrealsense2 as rs
import numpy as np
from enum import IntEnum
from matplotlib import pyplot as plt
from datetime import datetime
import open3d as o3d
import cv2
import cv2.aruco as aruco
import os
import keyboard
import copy
import pandas as pd

import vtk
from vtk.util import numpy_support


from numpy.linalg import inv
from preregistration import *


def invertMat(transformation):
    rot = transformation[:3, :3]
    
    
    t = transformation[:3, 3]
    rot_trans = rot.transpose()
    
    ##check orthogonality
    #rot_trans=np.linalg.inv(rot)
    #check = np.matmul(rot, rot_trans)
    #print(check)
    #almost
    
    mult = -1*(np.dot(rot_trans, t))
    #create empty
    inverse = np.zeros([4, 4])
    inverse[:3, :3] = rot_trans[:, :]
    inverse[:3, 3] = mult
    inverse[3, 3] = 1

    #print(transformation)
    #print(inverse)
    return(inverse)



def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
    
    

"""Load Data"""


#210512: 1-first Feedback
#target = o3d.io.read_point_cloud(r"C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\CaptureBackFrame_PLY77.ply")
#pre_reg = np.load(r"C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\preT77.npy")


# Source= PC from CAD
#CAD MODEL
source = o3d.io.read_point_cloud(
    r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\SpineModelKR_V10ACTUALmeters.PLY')


#Target= PC from RealSense
target = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210512 Debug Reg 2_feedback\CaptureBackFrameDEBUG_PLY30.ply')

pre_reg = np.load(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210512 Debug Reg 2_feedback\DebugpreT30.npy')


"""Kenan's Example"""

# a = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
# b = np.array([[0, 0, 0], [0, 10, 0], [0, 0, 10], [0, 10, 10]])

# transformation = initialAlignment(a, b)
# print(" transformation matrix")
# print(transformation)

# newA = []
# for i in a.tolist():
#     newI = np.array(i+[1])
#     ii = np.dot(transformation, newI)
#     newA.append(ii[:3])
    
# print("transformed A: it should be very very close to 'b'")
# print(newA)
# newA = np.asarray(newA)
# print("To proof it, diff: b-newA")
# print(np.sum(b-newA))
# print("so newA (transformed a equals to b. So my pre-reg is correct)")




"""See if the Inverse Matrix is Correct"""
"""Apply a transformation to a PC, and undo it with Inverse Transf. Matrix"""

#Get Arbitrary coordinate frame 

# mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
# T = np.eye(4)
# T[:3, :3] = mesh.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
# T[0, 3] = 1
# T[1, 3] = 1.3
# print(T)

# new_inv=invertMat(T)

# #transform mesh
# mesh_t = copy.deepcopy(mesh).transform(T)
# #check to see if it undoes
# mesh_inv = copy.deepcopy(mesh_t).transform(new_inv)
# o3d.visualization.draw_geometries([mesh, mesh_t])
# o3d.visualization.draw_geometries([mesh, mesh_t, mesh_inv])
# print("Here we see that the inverse matrix is correct, when we apply the original matrix, and the the inverse, it is at the origina spot")


"""Let's see if we can do the same just on the source using T transformation"""

# #CAD
# source_temp = copy.deepcopy(source)
# source_temp.transform(T)
# source_undo = copy.deepcopy(source_temp).transform(new_inv)
# source_temp.paint_uniform_color([0, 0, 1])
# source_undo.paint_uniform_color([1, 1, 0])
# o3d.visualization.draw_geometries([source, source_temp])
# o3d.visualization.draw_geometries([source_temp, source_undo])
# print("Here we see that it undoes it as well!")


"""Again but witht the preregistration"""
flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

T = pre_reg

T=np.matmul(flip_transform,T)
#left_right_hand = np.array([[1, 0, 0,0], [0, - 1, 0,0],[0,  0, 1, 0], [0, 0, 0, 1]])
#T = np.matmul(left_right_hand, pre_reg)
#print(T)
#new_inv = invertMat(T)

#CAD


# source = source.transform(flip_transform)
source_temp = copy.deepcopy(source)
"""Can we transform and untransform"""

# source_temp.transform(T)
# source_undo = copy.deepcopy(source_temp).transform(new_inv)
# source_temp.paint_uniform_color([0, 0, 1])
# source_undo.paint_uniform_color([1, 1, 0])
# o3d.visualization.draw_geometries([source, source_temp])
# o3d.visualization.draw_geometries([source_temp, source_undo])
# o3d.visualization.draw_geometries([source_temp, source_undo])
# print("Here we see that it undoes it AGAIN!")

"""Let's try to registration: source.transform(T) keep target the same"""
# source_temp.paint_uniform_color([0, 0, 1])
# source_temp.transform(new_inv)
# source_x=copy.deepcopy(source).transform(T)
# source_x.paint_uniform_color([0, 1, 0])
# target_temp = copy.deepcopy(target).transform(new_inv)
# target_temp.paint_uniform_color([1,0.8,0])
# #o3d.visualization.draw_geometries([source_temp, target])
# o3d.visualization.draw_geometries(
#     [source_temp, source, target, source_x, target_temp])
# print("NO")

### Apply to Source###
#source_temp.paint_uniform_color([0, 0, 1]) #Blue
#source_temp.transform(new_inv) # Apply inverse T to source
source_x = copy.deepcopy(source).transform(T)  #Apply  T to source
source_x.paint_uniform_color([0, 1, 0]) #Green

### Apply to Target###
#target_temp = copy.deepcopy(target).transform(new_inv)
#target_temp.paint_uniform_color([1, 0, 0])
#target_x = copy.deepcopy(target).transform(T)
#target_x.paint_uniform_color([0, 0.5, 0])
#o3d.visualization.draw_geometries([source_temp, target])
#o3d.visualization.draw_geometries([source_temp, source, target, source_x, target_temp,target_x])
o3d.visualization.draw_geometries([source_x, source, target])



#http: // www.open3d.org/docs/0.11.0/tutorial/pipelines/icp_registration.html
#http://www.open3d.org/docs/0.11.0/tutorial/pipelines/global_registration.html
#http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html?highlight=transform#open3d.geometry.PointCloud.transform

print("review the steps here!")
print("Here in the global and local alignmnet the souce is.transformed()  ")
print("The transformation is then to get the source into the target frame")


print("It says about ICP")
"""      the input are two point clouds and an initial transformation 
      that roughly aligns the source point cloud to the target point cloud.
      The output is a refined transformation that tightly aligns the two point clouds"""
      
      
      
      
      
      
      

"""Evaluate the initial registration (alignment)"""
print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(
    source, target, 0.10, T)
print(evaluation)

reg_source=source.transform(T)

"""Calculate distance between transformed source to target"""
# Open3D provides the method compute_point_cloud_distance to compute the distance 
# from a source point cloud to a target point cloud. 
# I.e., it computes for each point in the source point cloud the distance to the 
# closest point in the target point cloud.

dist = reg_source.compute_point_cloud_distance(target)
dist = np.asarray(dist)*1000 # to get in [m]
#print(dist)
print(np.shape(dist))


dist_pd = pd.DataFrame(dist)
ax1 = dist_pd.boxplot(return_type="axes")  # BOXPLOT
ax2 = dist_pd.plot(kind="hist", alpha=0.5, bins=1000)  # HISTOGRAM
ax3 = dist_pd.plot(kind="line")  # SERIES
plt.show()




"""Then Apply p2p ICP"""
trans_init=T
threshold=10 #[m] so 
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)
o3d.visualization.draw_geometries(
    [copy.deepcopy(source).transform(reg_p2p.transformation), source, target])


"""Calculate distance between transformed source to target"""
















"""Let's try to registration: source.transform(new_inv) keep target the same"""
# source_OG = copy.deepcopy(source).paint_uniform_color([1, 0, 0])
# source_temp.paint_uniform_color([0, 0, 1])
# source_temp.transform(new_inv)
# o3d.visualization.draw_geometries([source, target])
# o3d.visualization.draw_geometries([target, source_temp, source_OG])
# # print("NO")

"""Let's try to registration: target.transform(new_inv) keep target the same"""
# source_temp.paint_uniform_color([1, 0, 0])
# #source_temp.transform(T)
# target_temp = copy.deepcopy(target).transform(new_inv)
# target_temp.paint_uniform_color([1, 1, 0])

# o3d.visualization.draw_geometries([source_temp, target_temp])
# o3d.visualization.draw_geometries([source_temp,target_temp,target])
# # print("NO")


"""BOTH T"""
# source_OG = copy.deepcopy(source).paint_uniform_color([1, 0, 0])
# source_temp.paint_uniform_color([0, 0, 1])
# source_temp.transform(T)
# target_temp = copy.deepcopy(target).transform(T)
# o3d.visualization.draw_geometries([source_temp, target_temp])
# o3d.visualization.draw_geometries([source_temp, target_temp,target,source_OG])
# # print("NO")

