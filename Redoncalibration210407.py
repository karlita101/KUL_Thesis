############## Camera Calibration##########################

import pyrealsense2 as rs
import numpy as np
import cv2
import glob
import keyboard 


def drawchessboard(square_size, width, height,dirpath, prefix, image_format):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepares object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((width*height,3), np.float32)
    objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2)

    objp = objp * square_size 

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    stat_images = glob.glob(dirpath+'/' + prefix + '*.' + image_format)
    ##print(len(stat_images))
    #Start counter for images used
    found_img=0
    for fname in stat_images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        print(ret)

        # If found, add object points, image points (after refining them)
        if ret == True: #ret or retval is a variable used to assign the status of the last executed command to the retval variable
            objpoints.append(objp)
            
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            found_img+=1 # count +1 when corneres are found in each frame
            
            cv2.imshow ('Image', img)
            cv2.waitKey(500)
        
    print("Number of images used for calibration: ", found_img)
    #cv2.waitKey(2)
    cv2.destroyAllWindows()
    #Calibration
    print("Start calibration")
    # Get camera matrix, distortion coefficients, rotation and translation vector.  
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Calibration done")
    #undistort (Added 28.10.2020)
    undistort(stat_images,mtx,dist)
    return [rms, mtx, dist, rvecs, tvecs, stat_images]

    #cv2.destroyAllWindows()

def save_coefficients(mtx, dist, tvecs, rvecs, dirpath):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(dirpath+'/'+'UpdateExtrinsicscalibration.txt', cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.write("R", rvecs)
    cv_file.write("T", tvecs)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def undistort(stat_images,mtx,dist):
    count=1
    for fname in stat_images:
        img = cv2.imread(fname)
        #3 Get dimensions of image: get width and heigh
        h,w=img.shape[:2] 
        
        ### overview of parameters
        #ROI: Rectongular region of interest in Open CV
        # cv.getOptimalNewCameraMatrix(	cameraMatrix, distCoeffs, imageSize, alpha[, newImgSize[, centerPrincipalPoint]])
        #alpha=1 when all the source image pixels are retained in the undistorted image). See stereoRectify for details.
        #alpha=1 means that the rectified image is decimated and shifted so that all the pixels from the original images 
        # from the cameras are retained in the rectified images (no source image pixels are lost). Any intermediate value yields an intermediate result between those two extreme cases.

        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        
        #undistort 
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        #crop the image
        
        #x,y are top left coordinates / #y+h are the bottom left coordinates / #x+w are the top right coordinates
        #Define the ROI 
        x,y,w,h = roi
        #crop image to only the specified bounds 
        dst = dst[y:y+h, x:x+w]
        countstr=str(count)
        cv2.imwrite('REDOcalibresult'+countstr+'.png',dst)
        count+=1


#Initialize directory path

#First calibration
dirpath = r'C:\Users\karla\OneDrive\Documents\GitHub\KUL_Thesis'


######################################################################################
# REDO EXECUTE CALIBRATION
#######################################################################################

#Get cameracalibration   
square_size =0.024
width=8
height=6
image_format='png'
prefix='Image'
rms, mtx, dist, rvecs, tvecs, stat_images= drawchessboard(square_size, width, height,dirpath, prefix, image_format)

#Save coefficients
save_coefficients(mtx, dist, tvecs, rvecs, dirpath)
