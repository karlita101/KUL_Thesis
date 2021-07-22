#Get Kuka arm to RS camera registration


import vtk
import numpy as np
from sourcetargetregistration import *
import json

def getinversetransform(T):
    Rmat = T[:3, :3]
    tvec = T[:3,3]
    Rtrans=np.transpose(Rmat)
    Rtranstvec=-1*(np.dot(Rtrans, tvec))

    #rodrigues to rot matrix
    T_mat = np.zeros((4, 4))
    T_mat[:3, :3] =Rtrans
    T_mat[:3, 3] = Rtranstvec
    T_mat[3, 3] = 1
    return T_mat


file_name =(r"C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210720 Evaluation Testing\cal_poses.json")
with open(file_name ) as json_file:
    data = json.load(json_file)
    Nlist = []
    
    for p in data['probe_frames']:
        cur= [p['x'],p['y'], p['z']]
        Nlist.append(cur)
                
#Target are the aruco marker positions XYZ  wrt to camera
#Source are the positions of the kuka end effector wrt to its coordinate system at the base

#source=kuka
target = np.load(r"C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210720 Evaluation Testing\2021-07-20_11_15_10_cali.npy")
#target = np.load(r"C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210720 Evaluation Testing\karla data\Calibration data\2021-07-20_11_15_10_cali.npy")
source=Nlist
source=np.array(source)
print("-------Nlist---------")
print(Nlist)
print(np.shape(Nlist))

print("-------Aruco Coordinates---------")
print(target)
print(np.shape(target))

registration = initialAlignment(source, target)
print("-------REG KUKA to ARUCO ----------")
print(registration)

#inv_trans=getinversetransform(registration)

