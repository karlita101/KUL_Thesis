#Jun 22, 2021
#Purpose: Create a pointcloud for tracking  path 

import numpy as np
import  open3d as  o3d
from preregistration import *
import copy


def gettrajectory(start_point, end_point, w, l,path):
    scan_start = start_point//(w+1)
    scan_end = end_point//(w+1)

    scan = scan_start
    initial_point = start_point

    trajectory = []

    while (scan <= scan_end) and (scan >= scan_start):
        print("initial_point", initial_point)
        print("scan", scan)

        #End value to scan to for each scan line
        val_end = (scan+1)*(w+1)
        print('val end', val_end)
        #if end value is within the same scan line
        if (scan < (val_end)//(w+1)) and (end_point < val_end):
            traj = path[initial_point:end_point+1]
            print(traj)
            print("Partial line")
        #complete the whole scan line, and move on
        else:
            traj = path[initial_point:val_end]
            print(traj)
            print("Whole line")
        #add grid path to global trajectory
        np.append(trajectory, traj)
        #update intial point to start off on the next interation
        initial_point = (scan+1)*(w+1)
        #next scan line
        scan += 1






source = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\SpineModelKR_V12UpperSurface.PLY')


#Target= PC from RealSense
#"C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210623PilotTestInvestigatePC"
#target = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210517PilotTest\pointclouds\BackPLY50.ply')
#target = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210623PilotTestInvestigatePC\pointclouds\BackPLY1695.ply')
target = o3d.io.read_point_cloud(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210624PilotTestAngles60\Angle30\pointclouds\BackPLY2443.ply')  # aruco = np.load(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210623PilotTestInvestigatePC\arucotvec\id_tvec1695.npy')
aruco = np.load(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210624PilotTestAngles60\Angle30\arucotvec\id_tvec2443.npy')

#pre_reg = np.load(r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210623PilotTestInvestigatePC\preregmat\preregT1695.npy')
pre_reg = np.load(
    r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210624PilotTestAngles60\Angle30\preregmat\preregT2443.npy')


print(aruco)

#Define  3 aruco marker to be origin
p2=aruco[0]
p3=aruco[2]
p1=aruco[3]

num=4
p4=aruco[0:num]
p4.reshape((num, 3))

print("ARUCO  Marker")
print(aruco[0:num])

w=2
l=3

path=[]
for i in range(l+1):
    for j in range(w+1):
        #print("i",i,"j",j)
        path.append(p1+i/l*(p3-p1)+j/w*(p2-p1))
print("path vs aruco")
#print(path)        

#Check corespondances to reported aruco positions       

print("Aruco 0", p2)
print("p2", path[2])

print("Aruco 1", aruco[1])
print("Position M1 extrapolated", path[11])

print("Aruco 2", p3)
print("p3", path[9])

print("Aruco 3", p1)
print("p1", path[0])

normal=np.cross((p3-p1),(p2-p1))

# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
path_pc = o3d.geometry.PointCloud()
path_pc.points = o3d.utility.Vector3dVector(path)
#o3d.io.write_point_cloud("../../test_data/sync.ply", pcd)


flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

fourth_pc= o3d.geometry.PointCloud()
fourth_pc.points = o3d.utility.Vector3dVector(p4)
fourth_pc.paint_uniform_color([0,0,1])
fourth_pc.transform(flip_transform)

#Colors
path_pc.paint_uniform_color([1.0, 0.0, 1.0])
#calculate Normals
#enter 'n' in keyboard to see normals
path_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
path_pc.transform(flip_transform)

#Visualize
#o3d.visualization.draw_geometries([path_pc])
o3d.visualization.draw_geometries([target])
o3d.visualization.draw_geometries([target, fourth_pc])
o3d.visualization.draw_geometries([target, fourth_pc, path_pc])

dist_path = path_pc.compute_point_cloud_distance(target)
dist_path = np.asarray(dist_path)*1000  # to get in [mm]
print("mean distance", np.mean(dist_path))


print("")
print("1) Please pick at least three correspondences using [shift + left click]")
print("   Press [shift + right click] to undo point picking")
print("2) Afther picking points, press q for close the window")
vis = o3d.visualization.VisualizerWithEditing()
vis.create_window()
vis.add_geometry(target)
vis.run()  # user picks points
vis.destroy_window()
print("")


start_point=4
end_point=7

trajectory = gettrajectory(start_point, end_point, w, l,path)

