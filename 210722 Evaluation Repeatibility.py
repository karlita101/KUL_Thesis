#Repeatibility Study
#Using 5 trials of the calibration JSON files

import numpy as np
from glob import glob
from sourcetargetregistration import *
import json
import vtk

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

file_names = glob(
    './210720 Evaluation Testing/karla data/Calibration_Precision data/*')

#Load all arrays (trials, 4, 3)
#DIVIDE by 100) to get in METERS
arrays = [np.load(f)/1000 for f in file_names]

print("arrays being used")
print(arrays)

reg_arr = [initialAlignment(source, arr) for arr in arrays]


print("----- GET REGISTRATION MATRICES---")
print(np.shape(reg_arr))
#print(reg_arr)
reg_arr = np.reshape(reg_arr, (5, 4, 4))

print("----- LOAD Calibration Registration Matrix-------")
regMat= np.load('./210720 Evaluation Testing/Regmat_kuka2aruco_in_m_withoutprecision.npy')
print(regMat)


"""Stats of rotation"""
#Test

# print(reg_arr[:, 0, 0])
ave_R00 = np.mean(np.array(reg_arr[:,0,0]), axis=0)
std_R00 = np.std(reg_arr[:, 0, 0], axis=0)

ave_R01 = np.mean(reg_arr[:, 0, 1], axis=0)
std_R01 = np.std(reg_arr[:, 0, 1], axis=0)

ave_R02 = np.mean(reg_arr[:, 0, 2], axis=0)
std_R02 = np.std(reg_arr[:, 0, 2], axis=0)


ave_R10 = np.mean(reg_arr[:, 1, 0], axis=0)
std_R10 = np.std(reg_arr[:, 1, 0], axis=0)

ave_R11 = np.mean(reg_arr[:, 1, 1], axis=0)
std_R11 = np.std(reg_arr[:, 1, 1], axis=0)

ave_R12 = np.mean(reg_arr[:, 1, 2], axis=0)
std_R12 = np.std(reg_arr[:, 1, 2], axis=0)


ave_R20 = np.mean(reg_arr[:, 2, 0], axis=0)
std_R20 = np.std(reg_arr[:, 2, 0], axis=0)

ave_R21 = np.mean(reg_arr[:, 2, 1], axis=0)
std_R21 = np.std(reg_arr[:, 2, 1], axis=0)

ave_R22 = np.mean(reg_arr[:, 2, 2], axis=0)
std_R22 = np.std(reg_arr[:, 2, 2], axis=0)


"""Stats of translation"""


ave_Tx = np.mean(reg_arr[:, 0, 3], axis=0)
std_Tx = np.std(reg_arr[:, 0, 3], axis=0)


ave_Ty = np.mean(reg_arr[:, 1, 3], axis=0)
std_Ty = np.std(reg_arr[:, 1, 3], axis=0)


ave_Tz = np.mean(reg_arr[:, 2, 3], axis=0)
std_Tz = np.std(reg_arr[:, 2, 3], axis=0)


# ave_1= np.mean(reg_arr[:, 3, 3], axis=0)
# std_1 = np.std(reg_arr[:, 3, 3], axis=0)

# print(ave_1)

print("mean marix")
mean_mat = [[ave_R00, ave_R01, ave_R02, ave_Tx], [
    ave_R10, ave_R11, ave_R12, ave_Ty], [ave_R20, ave_R21, ave_R22, ave_Tz]]

mean_mat=np.array(mean_mat)
print(mean_mat)

print("std marix")
std_mat = [[std_R00, std_R01, std_R02, std_Tx], [
    std_R10, std_R11, std_R12, std_Ty], [std_R20, std_R21, std_R22, std_Tz]]

std_mat = np.array(std_mat)
print(std_mat)


#BIAS
#HOW far the ground truth is from the MEAN
print("-------BIAS Measurement---------")
bias = regMat[:3, :]-mean_mat
print(bias)

print("check")
print(regMat[:3, :])
print(mean_mat)

print("-------Total Analytical Error---------")
TAE=bias+2*std_mat
print(TAE)

print("std")
print(std_mat)
