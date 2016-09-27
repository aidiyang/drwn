import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numpy.linalg import norm
import py_utils as util

if len(sys.argv) < 2:
    print "python [data.csv]"
    sys.exit

f = sys.argv[1]

#df = pd.read_csv(f, sep=',')

#t = df['time']
#ctrl = df.filter(regex='ctrl').values
#conf = df.filter(regex='conf').values
#est_snsr = df.filter(regex='est_s').values
#snsr = df.filter(regex='snsr').values
#print snsr.shape
#
#
#if np.all(est_snsr):
#    snsr = est_snsr
#
#mrkr = np.copy(snsr[:,58:])
#print mrkr.shape, len(t)
#if (mrkr.shape[1] > 107):
#    mrkr = mrkr.reshape(len(t), 16,4)
#    print mrkr[0,:,3]
#else:
#    mrkr = mrkr.reshape(len(t), 16,3)
#ps = np.copy(mrkr[:,:,0:3])

hw = util.get_real_data(f, -1)

mrkr =hw['ps']
t = hw['time']
conf = hw['conf']

#if np.any(conf):
#    conf = np.ones((len(t), 16)) * 10 # full confidence

print mrkr[0,:,:], conf[0,:]


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

h = 200
l = 3

#print np.max(t)
#bools = np.vstack((conf[:,0]>l, conf[:,0]<h, t[:]>7))
#print bools
#bools = np.all(bools, axis=0)
#print bools.shape
#print bools

dt_tol=1000.001
#for i in range(0,16):
#    bools = np.all(np.vstack((conf[:,i]>l, conf[:,i]<h)), axis=0)
#    x = mrkr[bools,i,0]
#    y = mrkr[bools,i,1]
#    z = mrkr[bools,i,2]
#
#    nd = False
#    #for t in range(0,5):
#    c = 0
#    while np.all(nd) == False:
#        dx = np.ediff1d(x)
#        dy = np.ediff1d(y)
#        dz = np.ediff1d(z)
#
#        sx = np.square(dx)
#        sy = np.square(dy)
#        sz = np.square(dz)
#
#        d = sx+sy+sz
#        nd = np.sqrt(d)
#        #print nd.shape
#        #print np.max(nd)
#        #print nd[:]<dt_tol
#
#        bools = nd[:]<dt_tol
#        x = x[bools]
#        y = y[bools]
#        z = z[bools]
#        c= c+1
#    
#    print i, " took ", c, " iterations"
#    ax.plot(x, y, zs=z, marker='.')

conf_cutoff = 3
vel_limit = 0.1
min_time = 2
new_c = util.clean_mrkr_data(t, mrkr, conf, conf_cutoff, vel_limit, min_time)

for i in range(0,16):
    #bools = np.all(np.vstack((new_c[:,i]>l, new_c[:,i]<h)), axis=0)
    bools = new_c[:,i]>l
    #bools = np.any(new_c[:,i]>l) 
    x = mrkr[bools,i,0]
    y = mrkr[bools,i,1]
    z = mrkr[bools,i,2]
    
    print i, "num good:", x.shape 
    if (z.shape[0] > bools.shape[0]/2):
        print '\tused', i
        ax.plot(x, y, zs=z, marker='.')

#ax.set_xlim3d(-0.06, 0.08)
#ax.set_ylim3d(-.15, .15)
#ax.set_zlim3d(0, 0.4)

plt.show()
