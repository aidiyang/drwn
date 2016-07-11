import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numpy.linalg import norm

if len(sys.argv) < 2:
    print "python [data.csv]"
    sys.exit

f = sys.argv[1]

df = pd.read_csv(f, sep=',')

t = df['time']
ctrl = df.filter(regex='ctrl').values
snsr = df.filter(regex='snsr').values
conf = df.filter(regex='conf').values
print snsr.shape

#qpos = snsr[:,0:20]
#qvel = snsr[:,20:40]
#accl = snsr[:,40:43]
#gyro = snsr[:,43:46]
#ctct = snsr[:,46:58]

mrkr = snsr[:,58:]
if (mrkr.shape[1] > 107):
    mrkr = mrkr.reshape(len(t), 16,4)
    print mrkr[0,:,3]
else:
    mrkr = mrkr.reshape(len(t), 16,3)
#ps = np.copy(mrkr[:,:,0:3])


print mrkr[0,:,:], conf[0,:]


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

h = 10
l = 3

bools = np.vstack((conf[:,0]>l, conf[:,0]<h))
print bools.shape
bools = np.all(bools, axis=0)
print bools.shape

dt_tol=0.001
for i in range(0,16):
    bools = np.all(np.vstack((conf[:,i]>l, conf[:,i]<h)), axis=0)
    x = mrkr[bools,i,0]
    y = mrkr[bools,i,1]
    z = mrkr[bools,i,2]

    nd = False
    #for t in range(0,5):
    c = 0
    while np.all(nd) == False:
        dx = np.ediff1d(x)
        dy = np.ediff1d(y)
        dz = np.ediff1d(z)

        sx = np.square(dx)
        sy = np.square(dy)
        sz = np.square(dz)

        d = sx+sy+sz
        nd = np.sqrt(d)
        #print nd.shape
        #print np.max(nd)
        #print nd[:]<dt_tol

        bools = nd[:]<dt_tol
        x = x[bools]
        y = y[bools]
        z = z[bools]
        c= c+1
    
    print i, " took ", c, " iterations"
    ax.plot(x, y, zs=z, marker='.')

plt.show()
