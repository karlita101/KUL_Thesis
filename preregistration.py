import vtk
import numpy as np
def vtkmatrix_to_numpy(matrix):
    """
    Copies the elements of a vtkMatrix4x4 into a numpy array.

    :param matrix: The matrix to be copied into an array.
    :type matrix: vtk.vtkMatrix4x4
    :rtype: numpy.ndarray
    """
    m = np.ones((4, 4))
    for i in range(4):
        for j in range(4):
            m[i, j] = matrix.GetElement(i, j)
    return m

# Numpy array to vtk points
def numpyArr2vtkPoints(npArray):
    vtkpoints = vtk.vtkPoints()    
    for i in range(npArray.shape[0]):
        #print npArray[i]
        vtkpoints.InsertNextPoint(npArray[i])       
    return vtkpoints


def initialAlignment(source, target):
    # retrieval LM from mouse click 
    len_source  = source.shape[0]
    len_target = target.shape[0]
    
    #July 7/21 delete unstable aruco marker (MANUALLY DONE)
    marker=0
    stab_source=np.delete(np.copy(source),marker, axis=0)
    stab_target = np.delete(np.copy(target), marker, axis=0)

    if len_source == len_target and len_source !=0 and len_target!=0:
        #source[:3]
        tmpSource = numpyArr2vtkPoints(stab_source)
        tmpTarget = numpyArr2vtkPoints(stab_target)
        #check that it is a 3x3
        #print("----------"*10)
        #print(source[:3].shape)
       
        landmarkTransform = vtk.vtkLandmarkTransform()
        landmarkTransform.SetSourceLandmarks(tmpSource)
        landmarkTransform.SetTargetLandmarks(tmpTarget)
        landmarkTransform.SetModeToRigidBody()
        landmarkTransform.Update()

        matrixnp = vtkmatrix_to_numpy(landmarkTransform.GetMatrix())
        # print(matrixnp)
        return matrixnp


#a=source (from my camera)
#b=target (cad)
#a=np.array([[1,1,1],[2,2,2],[3,30,3],[4,40,4]])
#b=np.array([[12,3,15],[23,22,27],[35,2,31],[42,4,46]])

#res =initialAlignment(a,b)
#print(res)
