
import numpy as np\
import cv2
import itertools 

ids=[*range(1,4)]
print(ids)
#print(ids)
#comb=list(itertools.combinations(range(4), 2))
#print(comb)

#print(comb[0])

def combpairs(markerids):
    
    indices=[*range(len(markerids))]
    comb=list(itertools.combinations(range(len(markerids)), 2))
    print(comb)
    return comb

comb=combpairs(ids)
print('Done',comb)

for pairs in comb:
    print(pairs[0])
    print(pairs[1])
    


rmat=np.eye(3)
print(rmat)
rvec, _=cv2.Rodrigues(rmat)
print(rvec)