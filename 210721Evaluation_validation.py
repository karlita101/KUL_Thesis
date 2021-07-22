import numpy as np
from glob import glob
from sourcetargetregistration import *
import json
import vtk
import math

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


#Registration matrix from Kuka to ARUCO Marker 
#source=kuka
#target=aruco
#"C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210720 Evaluation Testing\Regmat_kuka2aruco.npy"
regmatrix = np.load('./210720 Evaluation Testing/Regmat_kuka2aruco2.npy')
print("-------reg matrix----------------------")
print(regmatrix)

Inv_regmatrix=getinversetransform((regmatrix))


print("------Inverse registration--------")
print(Inv_regmatrix)


print("-----------Test Inverse Regmatrix")
print(np.matmul(Inv_regmatrix,regmatrix))



#Load validation data 
#this is all aruco coordinates

file_names = glob('./210720 Evaluation Testing/karla data/Validation data/*')
print(len(file_names))


arrays = [np.load(f) for f in file_names]
print(np.shape(arrays))


print("-----ARUCO VALE TRIAL 1st-----")
print(arrays[0])
#get coordinates wrt to kuka coordinate
#Kuka coord= Tinv*aruco

kuka_equi=np.empty(np.shape(arrays))
print(np.shape(arrays)[0])

#only 3 effective trajectors 0-1-2-3
traj_equi = np.empty((4,3,1))

#kuka (ground truth trajectories)
traj_groundT = np.empty((3,  1))

for trial, arr in enumerate(arrays):
    print("---trial-----")
    print(trial)
    #print(np.shape(arr))
    #for each array (trial)
    for index, point in enumerate(arr):
        #index cuz we have 4 points for each trail
        print("-----Print point-----")
        #print(point)
        #print(np.shape(point))
        print(index)
                
        #Malt mult and get to 
        
        """point coordinates need to be in homogenous vector form"""
        homog_vec=np.append(point,1)
        #print(homog_vec)
        print(np.shape(homog_vec))
        
        #get transformed point coordinate
        row = np.matmul(Inv_regmatrix, homog_vec)
        print("-------Transformed aruco to KUKA coordinate-----")
        print(row)
        
        #only keep 0,1,2 = x,y,z coordinates values in the kuka arm reference frame
        kuka_equi[trial,index,:]=row[:3]
        

print("Now we have kuka equivalent")


"""Let's get the real Kuka coordinates"""


kuka_val_json = (
    r"C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210720 Evaluation Testing\cal_poses_validation.json")
with open(kuka_val_json) as json_file:
    data = json.load(json_file)
    Nlist = []

    for p in data['probe_frames']:
        cur = [p['x'], p['y'], p['z']]
        Nlist.append(cur)

Nlist = np.array(Nlist)

"""Print Nlist vs Aruco Derives wrt KUKA frame"""
print("Ground truth from JSON")
print(Nlist)
print("Measured (Aruco) Points wrt to kuka ONLY TRIAL 1")
print(kuka_equi[0,:,:])



#check tranjectory distances  
for trial, equiv in enumerate(kuka_equi):
    print("trial")
    print(trial)
    #trajectory 0-1
    traj01 = kuka_equi[trial, 1, :]-kuka_equi[trial, 0, :]
    traj01 = np.linalg.norm(traj01)
    
    #1-2
    traj12 = kuka_equi[trial, 2, :]-kuka_equi[trial, 1, :]
    traj12 = np.linalg.norm(traj12)
    #2-3
    traj23 = kuka_equi[trial, 3, :]-kuka_equi[trial, 2, :]
    traj23 = np.linalg.norm(traj23)
    
    traj_equi[trial, 0, :] = traj01
    traj_equi[trial, 1, :] = traj12
    traj_equi[trial, 2, :] = traj23
    
    
# aruco_traj01 = kuka_equi[0, 1, :]-kuka_equi[0, 0, :]
# aruco_traj01 = np.linalg.norm(aruco_traj01)
# print(aruco_traj01)

"""Get Ground truth trajectories"""
traj_groundT[0, :] = np.linalg.norm(Nlist[1, :]-Nlist[0, :])
traj_groundT[1, :] = np.linalg.norm(Nlist[2, :]-Nlist[1, :])
traj_groundT[2, :] = np.linalg.norm(Nlist[3, :]-Nlist[2, :])
    


print("-----Ground truth----")
print(traj_groundT)
print("---------Aruco Derived-----------")
print(traj_equi)

print("-----RMSE for trajectories----------")
# print(traj_groundT[1])
# print(traj_equi[:,1,:])

RMSE=np.empty((3,1))

for i in range(3):
    #traj equiv { every trial, trajectory= num , :}
    MSE_01 = np.square(np.subtract(traj_groundT[0], traj_equi[:, 0, :])).mean()
    RMSE[0] = math.sqrt(MSE_01)
    
    MSE_12 = np.square(np.subtract(traj_groundT[1], traj_equi[:, 1, :])).mean()
    RMSE[1] = math.sqrt(MSE_12)
    
    MSE_23 = np.square(np.subtract(traj_groundT[2], traj_equi[:, 2, :])).mean()
    RMSE[2] = math.sqrt(MSE_23)

print("Root Mean Square Error:\n")
print(RMSE)


# kuka_traj01 = Nlist[1, :]-Nlist[0, :]
# kuka_traj01 = np.linalg.norm(kuka_traj01)
# print(kuka_traj01)


