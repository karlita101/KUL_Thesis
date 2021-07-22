import numpy as np
from glob import glob
from sourcetargetregistration import *
import json
import vtk

"""Get Kuka JSON Positions: Source"""
kuka_reg_json = (r"C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis\210720 Evaluation Testing\cal_poses_calibrationdata_.json")
with open(kuka_reg_json) as json_file:
    data = json.load(json_file)
    Nlist = []

    for p in data['probe_frames']:
        cur = [p['x'], p['y'], p['z']]
        Nlist.append(cur)
        
Nlist=np.array(Nlist)

""""Get Recorded Aruco Positions"""
#First checked that all matrices  were of shape (4,3) no points missing
#Also manually checked that the values were within the same values

file_names = glob('./210720 Evaluation Testing/karla data/Calibration data/*')

print(len(file_names))
count = 0
for f in file_names:
    arr = np.load(f)
    #manually check values
    # print("-----ARRAY-----")
    # print("COUNT", count)
    # print(arr)
    
    if np.shape(arr)==(4,3):
        if count==0:
            #initalize the array
            target=arr
            source=Nlist
          
        else:
            #stack_arr=np.concatenate((stack_arr, arr), axis=0)
            target = np.vstack((target, arr))
            source = np.vstack((source, Nlist))
        #print("TRUE")
        count += 1
        #print(arr)
    
""""Check That Dimentions correspond"""
print("-----shape of Target array-------")
print(np.shape(target))
print("-----shape of Source array-------")
print(np.shape(source))
    
"""Get Registration Matrix"""
#both have shape (128,3)
registration = initialAlignment(source, target)
print("-------REG KUKA to ARUCO ----------")
print(registration)


""""Save Registration Matrix"""
np.save('./210720 Evaluation Testing/Regmat_kuka2aruco2',registration)
#np.save('./210624PilotTestAngles60/Angle30/arucotvec/'+'id_tvec'+str(frame_count), id_tvec)
