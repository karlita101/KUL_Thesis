import numpy as np
import os, sys
import pandas as pd
import glob

from matplotlib import pyplot as plt


#absolute directory
dir = os.path.abspath(".")


"""Evaluate geometric distances"""
#Distance Folder 
folder_dist = './210517PilotTest/distancesnpy'

arrays = {}

file_paths = glob.glob(os.path.join(folder_dist, '*.npy'))
# keys are the filenames
array_dict={os.path.basename(f)[-7:-4]: np.load(f) for f in file_paths}
print(array_dict)


d_pd=pd.DataFrame.from_dict(array_dict, orient='index',
                       columns=['UL-UR', 'UR-LR'])

print(d_pd)

ax1 = d_pd.boxplot(return_type="axes")  # BOXPLOT
ax2 = d_pd.plot(kind="hist", alpha=0.5, bins=1000)  # HISTOGRAM
ax3 = d_pd.plot(kind="line")  # SERIE
plt.show()


"""Evaluate aruco coordinates"""

#Distance Folder 
folder_dist = './210517PilotTest/distancesnpy'