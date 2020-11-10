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
    R, _ = cv2.Rodrigues(rvec)
    #Need to calculate the inverse. Note that the inverse of a rotation matrix is its transpose
    invRot=np.matrix(R).T
    invTvec= -(np.dot(invRot,np.matrix(-Tvec)) 
    #Return inverse rotation vector
    invRvec , _ = cv2.Rodrigues(invRot)
    return invRvec, invTvec

def relativePose(Rvec1,Rvec2,tvec1,tvec2):
    #Transfomration from marker 1 to marker 2 is given by 
    # Transfromration_M1tocamer(Rvec1,tvec1) * (multiplied) by inverse(Transformation_M2tocamera(Rvec2,tvec2))
    
    #Source 1 Recommends reshaping
    Rvec1, tvec1 = Rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    Rvec2, tvec2 = Rvec2.reshape((3, 1)), tvec2.reshape((3, 1)
                                                        
    invRvec2, invTvec2 = homogeneousInvTransform(Rvec2,tvec2)
    
    #Use cv2.composer RT function to compite two rotation and shift transformations
    transform =cv2.composeRT(Rvec1,tvec1,invRvec2,invTvec2)
    #reshape
    R_rel=transform[0]
    t_rel=transform[1]
    
    R_rel = R_rel.reshape((3, 1))
    t_rel = t_rel.reshape((3, 1))
    
    return R_rel, t_rel 
