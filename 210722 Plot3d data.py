import numpy as np
from glob import glob
from sourcetargetregistration import *
import json
import vtk

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def getinversetransform(T):
    Rmat = T[:3, :3]
    tvec = T[:3, 3]
    Rtrans = np.transpose(Rmat)
    Rtranstvec = -1*(np.dot(Rtrans, tvec))

    #rodrigues to rot matrix
    T_mat = np.zeros((4, 4))
    T_mat[:3, :3] = Rtrans
    T_mat[:3, 3] = Rtranstvec
    T_mat[3, 3] = 1
    return T_mat


def getkukaequivalent(regmatrix, file_names):
    print(regmatrix)
    Inv_regmatrix = getinversetransform((regmatrix))
    arrays = [np.load(f) for f in file_names]
    kuka_equi = np.empty(np.shape(arrays))
    print("------NUMBER OF TRIALS-------")
    print(np.shape(arrays)[0])

    for trial, arr in enumerate(arrays):
        print("---trial-----")
        print(trial)
        #print(np.shape(arr))
        #for each array (trial)
        for index, point in enumerate(arr):
            #index cuz we have 4 points for each trail
            print(index)
            print("-----Print point-----")
            # print(point)
            # print(np.shape(point))

            # /1000 to get into m !!!
            print("----get in m!------")
            point = point/1000
            #print(point)

            """point coordinates need to be in homogenous vector form"""
            homog_vec = np.append(point, 1)
            #print(homog_vec)
            print(np.shape(homog_vec))

            ##Malt mult and get transformed point coordinate
            row = np.matmul(Inv_regmatrix, homog_vec)
            print("-------Transformed aruco to KUKA coordinate-----")
            print(row)

            #only keep 0,1,2 = x,y,z coordinates values in the kuka arm reference frame
            kuka_equi[trial, index, :] = row[:3]
    return kuka_equi


regmatrix = np.load('./210720 Evaluation Testing/Regmat_kuka2aruco_in_m_withoutprecision.npy')
file_names_val = glob('./210720 Evaluation Testing/karla data/Validation data/*')

kuka_equiv_validation=getkukaequivalent(regmatrix,file_names_val)


file_names_calib = glob('./210720 Evaluation Testing/karla data/Calibration data/*')
kuka_equiv_calibration = getkukaequivalent(regmatrix, file_names_calib)


"""Get Kuka JSON Positions: Source"""
kuka_reg_json = (r"C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210720 Evaluation Testing\cal_poses_calibrationdata_.json")
with open(kuka_reg_json) as json_file:
    data = json.load(json_file)
    Nlist = []

    for p in data['probe_frames']:
        cur = [p['x'], p['y'], p['z']]
        Nlist.append(cur)
        
Nlist=np.array(Nlist)
print("------JSON-----")
print(Nlist)



kuka_val_json = (r"C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210720 Evaluation Testing\cal_poses_validation.json")
with open(kuka_val_json) as json_file:
    data = json.load(json_file)
    Nlist_val = []

    for p in data['probe_frames']:
        cur = [p['x'], p['y'], p['z']]
        Nlist_val.append(cur)
        
Nlist_val=np.array(Nlist_val)

print("----Done loading JSON-----")


fig = plt.figure()
labels = ["Kuka Calibration JSON Coordinates", "Kuka Validation JSON Coordinates",
          "Measured Aruco Validation Coordinates", "Measured Aruco Calibration Coordinates"]
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Measured Aruco and Kuka Positions in Kuka Reference Frame ")
#calibration JSON
ax.scatter(Nlist[:, 0], Nlist[:, 1], Nlist[:, 2], c="red", label=labels[0])
#validation JSON
ax.scatter(Nlist_val[:, 0], Nlist_val[:, 1],
           Nlist_val[:, 2], c="blue", label=labels[1])

ax.scatter(kuka_equiv_validation[:,:, 0], kuka_equiv_validation[:,:, 1],
           kuka_equiv_validation[:, :, 2], marker="x", c="orange", label=labels[2])

ax.scatter(kuka_equiv_calibration[:, :, 0], kuka_equiv_calibration[:, :, 1],
           kuka_equiv_calibration[:, :, 2], marker="x", c="green", label=labels[3])

ax.set_xlabel("X [m]")

ax.set_ylabel("Y [m]")

ax.set_zlabel("Z [m]")


ax.legend(loc="best")

plt.show()
