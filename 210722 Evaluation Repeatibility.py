#Repeatibility Study
#Using 5 trials of the calibration JSON files

import numpy as np
from glob import glob
from sourcetargetregistration import *
import json
import vtk
import math

def getrollpitchya(Rzyx):
    #Return roll, pitch, ya w in DEGREES
    roll = math.atan2(Rzyx[2,1],Rzyx[2,2])* 180 / math.pi
    pitch = math.atan2(Rzyx[2, 0], math.sqrt(math.pow(Rzyx[2, 1], 2)+math.pow(Rzyx[2, 2], 2)))* 180 / math.pi
    yaw = math.atan2(Rzyx[1, 0], Rzyx[0, 0]) * 180 / math.pi
       
    return  roll, pitch, yaw
    
"""Get Kuka JSON Positions: Source"""
kuka_reg_json = (
    r"C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210720 Evaluation Testing\cal_poses_calibrationdata_.json")
with open(kuka_reg_json) as json_file:
    data = json.load(json_file)
    Nlist = []

    for p in data['probe_frames']:
        cur = [p['x'], p['y'], p['z']]
        Nlist.append(cur)

Nlist = np.array(Nlist)
source=Nlist

""""Get Recorded Aruco Positions"""

file_names = glob('./210720 Evaluation Testing/karla data/Calibration_Precision data/*')

#Load all arrays (trials, 4, 3)
#DIVIDE by 100) to get in METERS
arrays = [np.load(f)/1000 for f in file_names]

print("arrays being used")
print(arrays)

reg_arr = [initialAlignment(source, arr) for arr in arrays]


print("----------------------- GET REGISTRATION MATRICES-----------------")
print(np.shape(reg_arr))
#print(reg_arr)
reg_arr = np.reshape(reg_arr, (5, 4, 4))

print("----- LOAD Calibration Registration Matrix-------")
regMat= np.load('./210720 Evaluation Testing/Regmat_kuka2aruco_in_m_withoutprecision.npy')
print("----- Calibration Registration Matrix-------")
print(regMat)
print("-----  Calibration Registration Roll, Pitch, Yaw in Degrees-------")
groundT_angles=getrollpitchya(regMat[:3,:3])
print(groundT_angles)

#MM!!!!!!
print("-----  Calibration Registration Translation in mm-------")
groundT_translation = regMat[:3, 3]*1000
print(groundT_translation)

"""Mean of  rotation matrix elements"""
ave_R00 = np.mean((reg_arr[:,0,0]), axis=0)
ave_R01 = np.mean(reg_arr[:, 0, 1], axis=0)
ave_R02 = np.mean(reg_arr[:, 0, 2], axis=0)
ave_R10 = np.mean(reg_arr[:, 1, 0], axis=0)
ave_R11 = np.mean(reg_arr[:, 1, 1], axis=0)
ave_R12 = np.mean(reg_arr[:, 1, 2], axis=0)
ave_R20 = np.mean(reg_arr[:, 2, 0], axis=0)
ave_R21 = np.mean(reg_arr[:, 2, 1], axis=0)
ave_R22 = np.mean(reg_arr[:, 2, 2], axis=0)


"""Mean and STD for roll, pitch, yaw"""
angles = [getrollpitchya(trial[:3,:3]) for trial in reg_arr]
print(np.shape(angles))
angles_ave=np.mean(angles,axis=0)
angles_std=np.std(angles,axis=0)

print("----- Mean Roll, pitch, yaw in Degree---------")
print(angles_ave)
print("----- STD Roll, pitch, yaw in Degree---------")
print(angles_std)


"""Stats of translation"""
ave_Tx = np.mean(reg_arr[:, 0, 3], axis=0)
std_Tx = np.std(reg_arr[:, 0, 3], axis=0)


ave_Ty = np.mean(reg_arr[:, 1, 3], axis=0)
std_Ty = np.std(reg_arr[:, 1, 3], axis=0)


ave_Tz = np.mean(reg_arr[:, 2, 3], axis=0)
std_Tz = np.std(reg_arr[:, 2, 3], axis=0)


print("----- Mean Translations [mm]---------")
ave_Translation = np.array([ave_Tx, ave_Ty, ave_Tz])*1000
print(ave_Translation)

print("----- STD Translations [mm]---------")
std_Translation = np.array([std_Tx, std_Ty, std_Tz])*1000
print(std_Translation)


print("-------------Mean Transformation Matrix-----------------------")
mean_mat = [[ave_R00, ave_R01, ave_R02, ave_Tx], [
    ave_R10, ave_R11, ave_R12, ave_Ty], [ave_R20, ave_R21, ave_R22, ave_Tz]]

mean_mat=np.array(mean_mat)
print(mean_mat)


# print("--------- Mean Roll, Pitch, Yaw in Degrees---------------------")
# mean_roll, mean_pitch, mean_yaw = getrollpitchya(mean_mat[:3, :3])
# print(mean_roll, mean_pitch, mean_yaw)


# print("std marix")
# std_mat = [[std_R00, std_R01, std_R02, std_Tx], [
#     std_R10, std_R11, std_R12, std_Ty], [std_R20, std_R21, std_R22, std_Tz]]

# std_mat = np.array(std_mat)
# print(std_mat)


#BIAS
#HOW far the ground truth is from the MEAN
print("-------ANGLE BIAS Measurement in DEG---------")
bias_angles = groundT_angles-angles_ave
print(bias_angles)


print("-------Total Analytical Error Angles in DEG---------")
TAE_angles=bias_angles+2*angles_std
print(TAE_angles)


print("-------Translation BIAS Measurement in mm---------")
bias_trans = groundT_translation-ave_Translation
print(bias_trans)

print("-------Total Analytical Error Translation in mm---------")
TAE_trans = bias_trans+2*std_Translation
print(TAE_trans)





