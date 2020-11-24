
import numpy as np
import cv2
import itertools 


def combpairs(markerids):
    #create indices for the marker ids
    #output pairs of possible id marker combinations using the N choose K tool in itertools
    indices = [*range(len(markerids))]
    comb = list(itertools.combinations(range(len(markerids)), 2))
    print(comb)
    comb = [[*pairs] for pairs in comb]
    return comb

markerids=[*range(4)]
print(markerids)
comb=combpairs(markerids)

#results=[ [*pairs] for pairs in comb]
#print(results)

print(comb)
[print(pair[1]) for pair in results]
