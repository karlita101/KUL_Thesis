#Jun 22, 2021
#Purpose: Create a pointcloud for tracking  path 

import numpy as np
import  open3d as  o3d


source = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\SpineModelKR_V12UpperSurface.PLY')


#Target= PC from RealSense
target = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210517PilotTest\pointclouds\BackPLY18.ply')

aruco= np.load(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210517PilotTest\arucotvec\id_tvec18.npy')

print(aruco)

#Define  3 aruco marker to be origin
p1=aruco[0]
p3=aruco[2]
p2=aruco[3]

w=3
l=4

path=np.empty([w, l])
for i in range(w+1):
    for j in range(l+1):
        path[i,j]=p1+i/l*(p3-p1)+j/w*(p2-p1)
        
        
normal=np.cross((p3-p1),(p2-p1))


# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
path_pc = o3d.geometry.PointCloud()
path_pc.points = o3d.utility.Vector3dVector(path)
#o3d.io.write_point_cloud("../../test_data/sync.ply", pcd)


#Colors
path_pc.paint_uniform_color(1, 0, 1)
#caclualte Normals
path_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

#Visualize
o3d.visualization.draw_geometries([path_pc])
