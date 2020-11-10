import cv2
import numpy as np

#Input: Extrinsic parpameters to bring points from world to camer coordinate system
#   Rvec-Rotation vector
#   tvec-translation vector 

#Output:
#   Tranformation for the relative pose between marker two markers (from marker A to B)

#Sources: 
# 1: https://medium.com/@aliyasineser/calculation-relative-positions-of-aruco-markers-eee9cc4036e3
# 2: # 2: https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#composert

# References:
# Linear algebra tools: https://medium.com/@aliyasineser/calculation-relative-positions-of-aruco-markers-eee9cc4036e3

# Understsanding transform: https://math.stackexchange.com/questions/152462/inverse-of-transformation-matrix

def homogeneousInvTransform(rvec,tvec):
    #Input: pose info for marker B
    #Calculate Rotation Matrix R from Rodrigues angles
    R, __ = cv2.Rodrigues(rvec)
    #Need to calculate the inverse. Note that the inverse of a rotation matrix is its transpose
    invRot=np.matrix(R).T
    invTvec= (np.dot(invRot,np.matrix(-tvec))) 
    #Return inverse rotation vector
    inv_rvec , __ = cv2.Rodrigues(invRot)
    return inv_rvec, inv_tvec

def relativePose(rvec1,rvec2,tvec1,tvec2):
    #Transfomration from marker 1 to marker 2 is given by 
    # Transfromration_M1tocamer(Rvec1,tvec1) * (multiplied) by inverse(Transformation_M2tocamera(Rvec2,tvec2))
    
    #Source 1 Recommends reshaping
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1)
                                                        
    inv_rvec, inv_tvec = homogeneousInvTransform(rvec2,tvec2)
    
    #Use cv2.composer RT function to compite two rotation and shift transformations
    comp_pose  = cv2.composeRT(rvec1, tvec1, inv_rvec, inv_tvec)[:2]
    
    r_rel, trel = comp_pose[0], comp_pose[1]
    
    #reshape
    ##Rrel, = rel[0].reshape((3, 1))
    ##trel = rel[1].reshape((3, 1))

    return r_rel, t_rel  
