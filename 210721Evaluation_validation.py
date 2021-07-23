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


"""Registration matrix from Kuka to ARUCO Marker """
#source=kuka
#target=aruco
#"C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210720 Evaluation Testing\Regmat_kuka2aruco.npy"
regmatrix = np.load('./210720 Evaluation Testing/Regmat_aruco2kuka_in_m_withoutprecision.npy')
#regmatrix = np.load('./210720 Evaluation Testing/Regmat_kuka2aruco_in_m_withoutprecision.npy')
print("-------reg matrix----------------------")
print(regmatrix)


#Inv_regmatrix=getinversetransform((regmatrix))


""""Load Measured Validation Aruco Positions"""
file_names = glob('./210720 Evaluation Testing/karla data/Validation data/*')
print(len(file_names))

#note all in mm
arrays = [np.load(f) for f in file_names]
print(np.shape(arrays))


# print("-----ARUCO VALE TRIAL 1st-----")
# print(arrays[0])
# #get coordinates wrt to kuka coordinate
# #Kuka coord= Tinv*aruco

print("-------Transform measured ARUCO coordinates wrt to KUKA frame-------")

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
        print(index)
        print("-----Print point-----")
        # print(point)
        # print(np.shape(point))
        
        
        # /1000 to get into m !!!
        print("----get in m!------")
        point=point/1000
        
        
        """point coordinates need to be in homogenous vector form"""
        homog_vec=np.append(point,1)
        #print(homog_vec)
        print(np.shape(homog_vec))
        
        ##Malt mult and get transformed point coordinate
        row = np.matmul(regmatrix, homog_vec)
        print("-------Transformed aruco to KUKA coordinate-----")
        print(row)
        
        #only keep 0,1,2 = x,y,z coordinates values in the kuka arm reference frame
        kuka_equi[trial,index,:]=row[:3]
        

print("Now we have kuka equivalent")


"""Get the Ground truth VALIDATION Kuka coordinates"""
kuka_val_json = (r"C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210720 Evaluation Testing\cal_poses_validation.json")
with open(kuka_val_json) as json_file:
    data = json.load(json_file)
    Nlist = []

    for p in data['probe_frames']:
        cur = [p['x'], p['y'], p['z']]
        Nlist.append(cur)

Nlist = np.array(Nlist)

"""Print Nlist vs Aruco Derives wrt KUKA frame"""
print("Ground truth from JSON in mm")
print(Nlist*1000)
print("Measured (Aruco) Points wrt to kuka for trial 0 in mm")
print(kuka_equi[1,:,:]*1000)



#check tranjectory distances  
#euiv=4 point coordinates for each trial
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

"""Get Ground truth trajectories in METERS"""
traj_groundT[0, :] = np.linalg.norm(Nlist[1, :]-Nlist[0, :])
traj_groundT[1, :] = np.linalg.norm(Nlist[2, :]-Nlist[1, :])
traj_groundT[2, :] = np.linalg.norm(Nlist[3, :]-Nlist[2, :])
    

print("-----Ground truth in METERS----")
print(traj_groundT)

print("-----Ground truth in MILLIMETERS----")
print(traj_groundT*1000)

print("---------Aruco Derived in METERS-----------")
print(traj_equi)


print("---------Aruco Derived in MILLIMETERS-----------")
print(traj_equi*1000)

print("-----RMSE for trajectories----------")
RMSE=np.empty((3,1))


#traj equiv { every trial, trajectory= num , :}
# print(traj_equi[:, 0, :])
# print(np.subtract(traj_groundT[0], traj_equi[:, 0, :]))


MSE_01 = np.square(np.subtract(traj_groundT[0], traj_equi[:, 0, :])).mean()
RMSE[0] = math.sqrt(MSE_01)

MSE_12 = np.square(np.subtract(traj_groundT[1], traj_equi[:, 1, :])).mean()
RMSE[1] = math.sqrt(MSE_12)

MSE_23 = np.square(np.subtract(traj_groundT[2], traj_equi[:, 2, :])).mean()
RMSE[2] = math.sqrt(MSE_23)

print("Root Mean Square Error in METERS:\n")
print(RMSE)


print("Root Mean Square Error in Millimeters:\n")
print(RMSE*1000)



