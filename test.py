
import numpy as np
import cv2
import itertools 

markerids=[*range(1,4)]
print(markerids)
#print(ids)
#comb=list(itertools.combinations(range(4), 2))
#print(comb)

#print(comb[0])

def combpairs(markerids):
    
    indices=[*range(len(markerids))]
    comb=list(itertools.combinations(range(len(markerids)), 2))
    print(comb)
    if np.all(comb != None):
        y=[3,2,1,0]
    return comb,y 

ans, ques =combpairs(markerids)
print('answer', ans)

comb=combpairs(ids)
print('Done',comb)

for pairs in comb:
    print(pairs[0])
    print(pairs[1])
    


rmat=np.eye(3)
print('Rmat',rmat)
rvec, __=cv2.Rodrigues(rmat)
print('rotation vector',rvec)
Rot,__ =cv2.Rodrigues(rvec)
print('redo ro tation matrix',Rot)