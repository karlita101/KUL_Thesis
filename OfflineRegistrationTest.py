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



#Target= PC from RealSense
target = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210512 Debug Reg 2_feedback\CaptureBackFrameDEBUG_PLY30.ply')

pre_reg = np.load(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210512 Debug Reg 2_feedback\DebugpreT30.npy')


o3d.visualization.draw_geometries([target])



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
o3d.visualization.draw_geometries(
    [source_temp,reg_source,target])


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
fig1.savefig("./210512 Debug Reg 2_feedback/210519 Get BaselinePerformance/PreRegICP_DebugpreT30_V12UpperSurf.png", dpi=150)
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
    
fig2.savefig("./210512 Debug Reg 2_feedback/210519 Get BaselinePerformance/PreRegICP_DebugpreT30_V12UpperSurf_HIST_remesh.png", dpi=150)

plt.show()



