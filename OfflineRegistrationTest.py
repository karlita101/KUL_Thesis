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


"""Load Data"""


#210512: 1-first Feedback
#target = o3d.io.read_point_cloud(r"C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\CaptureBackFrame_PLY77.ply")
#pre_reg = np.load(r"C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\preT77.npy")


# Source= PC from CAD
#CAD MODEL {course/walls}
#source = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\SpineModelKR_V10ACTUALmeters.PLY')

#CAD MODEL {course/nowalls}
#source = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\SpineModelKR_V11RemoveWalls.PLY')

#Remeshed Source
#source = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\SpineModelKR_V11RemoveWalls_RemeshMidpoint.PLY')
#Upper surface
source = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\SpineModelKR_V12UpperSurface.PLY')

o3d.visualization.draw_geometries([source])

#Target= PC from RealSense
#target = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210512 Debug Reg 2_feedback\CaptureBackFrameDEBUG_PLY30.ply')
#target = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210623PilotTestInvestigatePC\pointclouds\BackPLY1318.ply')
target = o3d.io.read_point_cloud(
    r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210624PilotTestAngles60\Angle30\pointclouds\BackPLY2443.ply')

#pre_reg = np.load(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210512 Debug Reg 2_feedback\DebugpreT30.npy')
#pre_reg = np.load(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210623PilotTestInvestigatePC\preregmat\preregT1318.npy')
pre_reg = np.load(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210624PilotTestAngles60\Angle30\preregmat\preregT2443.npy')


assembly = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\SpineBack_Assembly_KR_rev2.PLY')

o3d.visualization.draw_geometries([assembly])



voxel_size=1/100 # 1cm
down_assembly = assembly.voxel_down_sample(voxel_size)
down_assembly.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=30))
o3d.visualization.draw_geometries([down_assembly])



""" Visualize the mock tool"""
#July 8
flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
tool = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\0708MockInstrument_rev1_PLY.PLY')
#tool.transform(flip_transform)
o3d.visualization.draw_geometries([tool])


#read tool mesh
tool_mesh = o3d.io.read_triangle_mesh(
    r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\0708MockInstrument_rev1_PLY.PLY')

o3d.visualization.draw_geometries([tool_mesh])










o3d.visualization.draw_geometries([target])


mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh_r = copy.deepcopy(mesh)
R = mesh.get_rotation_matrix_from_xyz((0, np.pi, np.pi))
R_trans=np.transpose(R)
print(R_trans)

mesh_r.rotate(R, center=(0, 0, 0))
o3d.visualization.draw_geometries([mesh, mesh_r,target])


PC_coordframe = o3d.geometry.TriangleMesh.create_coordinate_frame()
PC_coordframe_r = copy.deepcopy(mesh)
R = mesh.get_rotation_matrix_from_xyz((0, np.pi, np.pi))
PC_coordframe_r.rotate(R, center=(0, 0, 0))
o3d.visualization.draw_geometries([mesh, mesh_r, target])






assembly.paint_uniform_color([112/255, 136/255, 153/255])
o3d.visualization.draw_geometries([assembly])

"""Again but witht the preregistration"""

#flip normals of Pre-reg Transformation
T = pre_reg
flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
T=np.matmul(flip_transform,T)

#Source Copy (Orange)
source_temp = copy.deepcopy(source)
source_temp.paint_uniform_color([1, 0.706, 0])
#Source Copy for Pre-reg
source_x = copy.deepcopy(source).transform(T)  #Apply  T to source
source_x.paint_uniform_color([0, 0.651, 0.929])  # Green
o3d.visualization.draw_geometries([source_x, source_temp, target])

"""Evaluate the initial registration (alignment)"""
print("Initial alignment")
threshold_1=0.5 #[m]
evaluation = o3d.pipelines.registration.evaluate_registration(
    source_temp, target, threshold_1, T)
print("pre-registration eval",evaluation)

#http: // www.open3d.org/docs/0.11.0/tutorial/pipelines/icp_registration.html
#http://www.open3d.org/docs/0.11.0/tutorial/pipelines/global_registration.html
#http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html?highlight=transform#open3d.geometry.PointCloud.transform

# print("review the steps here!")
# print("Here in the global and local alignmnet the souce is.transformed()  ")
# print("The transformation is then to get the source into the target frame")


# print("It says about ICP")
# """      the input are two point clouds and an initial transformation 
#       that roughly aligns the source point cloud to the target point cloud.
#       The output is a refined transformation that tightly aligns the two point clouds"""
      
      
 

"""Calculate distance between transformed source to target"""
# Open3D provides the method compute_point_cloud_distance to compute the distance 
# from a source point cloud to a target point cloud. 
# I.e., it computes for each point in the source point cloud the distance to the 
# closest point in the target point cloud.

#reg_source = copy.deepcopy(source).transform(T)
reg_source = copy.deepcopy(source_x)

dist = reg_source.compute_point_cloud_distance(target)
dist = np.asarray(dist)*1000 # to get in [mm]
#print(dist)
# print(np.shape(dist))


# dist_pd = pd.DataFrame(dist)
# ax1 = dist_pd.boxplot(return_type="axes")  # BOXPLOT
# ax2 = dist_pd.plot(kind="hist", alpha=0.5, bins=1000)  # HISTOGRAM
# ax3 = dist_pd.plot(kind="line")  # SERIES
# plt.show()



"""Then Apply p2p ICP"""
trans_init=T
threshold = 10  # [m] 	Maximum correspondence points-pair distance
reg_p2p = o3d.pipelines.registration.registration_icp(
    source_temp, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
print("ICP evalutation",reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)

source_icp =copy.deepcopy(source_temp).transform(reg_p2p.transformation).paint_uniform_color([0, 0.651, 0.929])

o3d.visualization.draw_geometries([source_temp,reg_source,target])
o3d.visualization.draw_geometries([reg_source, target])

o3d.visualization.draw_geometries([source_icp,target])

"""Calculate distance between transformed source to target"""
# Open3D provides the method compute_point_cloud_distance to compute the distance
# from a source point cloud to a target point cloud.
# I.e., it computes for each point in the source point cloud the distance to the
# closest point in the target point cloud.


dist_icp = source_icp.compute_point_cloud_distance(target)
dist_icp = np.asarray(dist_icp)*1000  # to get in [mm]
# #print(np.shape(dist))

# dist_pd_icp = pd.DataFrame(dist_icp)
# ax1 = dist_pd_icp.boxplot(return_type="axes")  # BOXPLOT
# ax2 = dist_pd_icp.plot(kind="hist", alpha=0.5, bins=1000)  # HISTOGRAM
# ax3 = dist_pd_icp.plot(kind="line")  # SERIES
# plt.show()



#define number of rows and columns for subplots
nrow = 1
ncol = 2


# Get statistics
mean_prereg, mean_icp = np.mean(dist), np.mean(dist_icp)
print('Pre_reg Mean', mean_prereg)
print('ICP Mean', mean_icp)

std_prereg, std_icp = np.std(dist), np.std(dist_icp)
print('Pre_reg std', std_prereg)
print('ICP std', std_icp)

med_prereg, med_icp = np.median(dist), np.median(dist_icp)
print('Pre_reg med', med_prereg)
print('ICP med', med_icp)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr_preg = '\n'.join((
    r'$\mu=%.2f$' % (mean_prereg, ),
    r'$\mathrm{median}=%.2f$' % (med_prereg, ),
    r'$\sigma=%.2f$' % (std_prereg, )))

textstr_icp = '\n'.join((
    r'$\mu=%.2f$' % (mean_icp, ),
    r'$\mathrm{median}=%.2f$' % (med_icp, ),
    r'$\sigma=%.2f$' % (std_icp, )))




 
fig1, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
fig1.suptitle('Source to Target Point Cloud Distances')
ax1.boxplot(dist,showmeans=True)
ax2.boxplot(dist_icp, showmeans=True)


#Add stats
ax1.text(0.05, 0.95, textstr_preg, transform=ax1.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
ax2.text(0.05, 0.95, textstr_icp, transform=ax2.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)

for ax in fig1.get_axes():
    ax.set(ylabel='Source to Target Distances [mm]')  # xlabel='x-label'
#Add title
ax1.set_title('Pre-registration PC Distances')
ax2.set_title('ICP PC Distances')
#fig1.savefig("./210512 Debug Reg 2_feedback/210519 Get BaselinePerformance/PreRegICP_DebugpreT30_V12UpperSurf.png", dpi=150)
fig1.savefig(
    "./210624PilotTestAngles60/Angle30/EvaluationImg/PreRegICP_T2443_V12UpperSurf.png", dpi=150)
plt.show()



#Add density=1 to normalize bt the total number of count
n_bins=50
fig2, (ax3, ax4) = plt.subplots(1, 2, sharey=True, tight_layout=True)
fig2.suptitle('Source to Target Point Cloud Distances')
ax3.hist(dist, bins=n_bins)
ax4.hist(dist_icp, bins=n_bins)

ax3.text(0.05, 0.95, textstr_preg, transform=ax3.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
ax4.text(0.05, 0.95, textstr_icp, transform=ax4.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)

for ax in fig2.get_axes():
    ax.set(xlabel='Point Cloud Distances [mm]',ylabel='Frequency')  # 
#Add title
ax3.set_title('Pre-registration PC Distances')
ax4.set_title('ICP PC Distances')
    
fig2.savefig(
    "./210624PilotTestAngles60/Angle30/EvaluationImg/PreRegICP_T2443_V12UpperSurf_HIST.png", dpi=150)


plt.show()



## apply the same transformation to the assembly

#assembly_temp=copy.deepcopy(assembly)
assembly_temp=copy.deepcopy(down_assembly)

#dt0 = datetime.now()
assembly_icp =assembly_temp.transform(reg_p2p.transformation)
#process_time = datetime.now() - dt0
#print("FPS: " + str(1 / process_time.total_seconds()))
#o3d.visualization.draw_geometries([assembly, assembly_icp, target])
#o3d.visualization.draw_geometries([assembly_icp, target])
o3d.visualization.draw_geometries([assembly_icp])

o3d.visualization.draw_geometries([assembly_temp, reg_source, target])
