import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt

from numpy.linalg import norm

if len(sys.argv) < 3:
    print "python [data1.csv] [data2.csv]"
    sys.exit

f = sys.argv[1]
r = sys.argv[2]

df = pd.read_csv(f, sep=',')

a_t = df['time']
a_ctrl = df.filter(regex='ctrl').values
a_snsr = df.filter(regex='snsr').values

a_nu = a_ctrl.shape[1]
a_ns = a_snsr.shape[1]

a_qpos = a_snsr[:,0:20]
a_qvel = a_snsr[:,20:40]
a_accl = a_snsr[:,40:43]
a_gyro = a_snsr[:,43:46]
a_ctct = a_snsr[:,46:58]
a_mrkr = a_snsr[:,58:]
if (a_ns > 107):
    a_mrkr = a_mrkr.reshape(len(a_t), 16,4)
    print a_mrkr[0,:,3]
else:
    a_mrkr = a_mrkr.reshape(len(a_t), 16,3)
a_ps = np.copy(a_mrkr[:,:,0:3])

a_dt = np.ediff1d(a_t)
a_mean_dt = np.mean(a_dt)
print "B Mean dt:", a_mean_dt


df2 = pd.read_csv(r, sep=',')

b_t = df2['time']
b_ctrl = df2.filter(regex='ctrl').values
b_snsr = df2.filter(regex='snsr').values

print len(b_t)
start = 300
b_t = b_t[start:]
print b_t[start]
b_t = np.subtract(b_t, b_t[start])
b_ctrl = b_ctrl[start:, :]
b_snsr = b_snsr[start:, :]
print b_snsr.shape
print len(b_t)

b_nu = b_ctrl.shape[1]
b_ns = b_snsr.shape[1]
b_qpos = b_snsr[:,0:20]
b_qvel = b_snsr[:,20:40]
b_accl = b_snsr[:,40:43]
b_gyro = b_snsr[:,43:46]
b_ctct = b_snsr[:,46:58]
b_mrkr = b_snsr[:,58:]
print b_mrkr.shape
if (b_ns > 107):
    b_mrkr = b_mrkr.reshape(len(b_t), 16,4)
    print b_mrkr[0,:,3]
else:
    b_mrkr = b_mrkr.reshape(len(b_t), 16,3)
b_ps = np.copy(b_mrkr[:,:,0:3])

b_dt = np.ediff1d(b_t)
b_mean_dt = np.mean(b_dt)
print "B Mean dt:", b_mean_dt



############### plot

fig, axs = plt.subplots(6, 3, sharex=False)

my_ls = '--'
my_lw = 2
my_alpha = 0.5
#axs[0,0].plot(t, qpos, lw=my_lw, alpha=my_alpha)
#axs[0,0].set_title('jointpos')
##axs[0,0].set_ylim([-1, 1])
#
#axs[0,1].plot(t, qvel, lw=my_lw, alpha=my_alpha)
#axs[0,1].set_title('jointvel')
##axs[0,1].set_ylim([-1, 1])
#
#axs[0,2].plot(t, accl, lw=my_lw, alpha=my_alpha)
#axs[0,2].set_title('accel')
##axs[0,2].set_ylim([-1, 1])
#
#axs[1,0].plot(t, gyro, lw=my_lw, alpha=my_alpha)
#axs[1,0].set_title('gyro')
##axs[1,0].set_ylim([-1, 1])
#
#axs[1,1].plot(t, ctct, lw=my_lw, alpha=my_alpha)
#axs[1,1].set_title('contact')
##axs[1,1].set_ylim([-1, 1])

#axs[1,2].plot(t, ps[:,:,2], lw=my_lw, alpha=my_alpha)
#axs[1,2].set_title('raw z-axis')
#axs[1,2].set_ylim([-1, 1])

axs[0,0].plot(a_t, a_accl[:,0], color='red', lw=my_lw, alpha=my_alpha)
axs[0,1].plot(a_t, a_accl[:,1], color='red', lw=my_lw, alpha=my_alpha)
axs[0,2].plot(a_t, a_accl[:,2], color='red', lw=my_lw, alpha=my_alpha)
axs[0,0].set_title('a accel-x')
axs[0,1].set_title('a accel-y')
axs[0,2].set_title('a accel-z')

axs[0,0].plot(b_t, b_accl[:,0], lw=my_lw, alpha=my_alpha)
axs[0,1].plot(b_t, b_accl[:,1], lw=my_lw, alpha=my_alpha)
axs[0,2].plot(b_t, b_accl[:,2], lw=my_lw, alpha=my_alpha)
#axs[1,0].set_title('b accel-x')
#axs[1,1].set_title('b accel-y')
#axs[1,2].set_title('b accel-z')


#axs[3,0].plot(f_accl, sp_accl[:,0].imag, lw=my_lw, alpha=my_alpha)
#axs[3,0].set_ylim([-10, 10])
#axs[3,0].set_title('fft accel-x')
#axs[3,1].plot(f_accl, sp_accl[:,1].imag, lw=my_lw, alpha=my_alpha)
#axs[3,1].set_ylim([-10, 10])
#axs[3,1].set_title('fft accel-y')
#axs[3,2].plot(f_accl, sp_accl[:,2].imag, lw=my_lw, alpha=my_alpha)
#axs[3,2].set_ylim([-10, 10])
#axs[3,2].set_title('fft accel-z')

axs[1,0].plot(a_t, a_gyro[:,0], color='red', lw=my_lw, alpha=my_alpha)
axs[1,1].plot(a_t, a_gyro[:,1], color='red', lw=my_lw, alpha=my_alpha)
axs[1,2].plot(a_t, a_gyro[:,2], color='red', lw=my_lw, alpha=my_alpha)
axs[1,0].set_title('A gyro-x') 
axs[1,1].set_title('A gyro-y')
axs[1,2].set_title('A gyro-z')
axs[1,0].plot(b_t, b_gyro[:,0], lw=my_lw, alpha=my_alpha)
axs[1,1].plot(b_t, b_gyro[:,1], lw=my_lw, alpha=my_alpha)
axs[1,2].plot(b_t, b_gyro[:,2], lw=my_lw, alpha=my_alpha)
#axs[3,0].set_title('b gyro-x') 
#axs[3,1].set_title('b gyro-y')
#axs[3,2].set_title('b gyro-z')

axs[2,0].plot(a_t, a_ctct[:,0], color='red', lw=my_lw, alpha=my_alpha)
axs[2,1].plot(a_t, a_ctct[:,1], color='red', lw=my_lw, alpha=my_alpha)
axs[2,2].plot(a_t, a_ctct[:,2], color='red', lw=my_lw, alpha=my_alpha)
axs[2,0].set_title('A r ctct-x') 
axs[2,1].set_title('A r ctct-y')
axs[2,2].set_title('A r ctct-z')
axs[2,0].plot(b_t, b_ctct[:,0], lw=my_lw, alpha=my_alpha)
axs[2,1].plot(b_t, b_ctct[:,1], lw=my_lw, alpha=my_alpha)
axs[2,2].plot(b_t, b_ctct[:,2], lw=my_lw, alpha=my_alpha)
#axs[5,0].set_title('b r ctct-x') 
#axs[5,1].set_title('b r ctct-y')
#axs[5,2].set_title('b r ctct-z')

axs[3,0].plot(a_t, a_ctct[:,6], color='red', lw=my_lw, alpha=my_alpha)
axs[3,1].plot(a_t, a_ctct[:,7], color='red', lw=my_lw, alpha=my_alpha)
axs[3,2].plot(a_t, a_ctct[:,8], color='red', lw=my_lw, alpha=my_alpha)
axs[3,0].set_title('A l ctct-x') 
axs[3,1].set_title('A l ctct-y')
axs[3,2].set_title('A l ctct-z')
axs[3,0].plot(b_t, b_ctct[:,6], lw=my_lw, alpha=my_alpha)
axs[3,1].plot(b_t, b_ctct[:,7], lw=my_lw, alpha=my_alpha)
axs[3,2].plot(b_t, b_ctct[:,8], lw=my_lw, alpha=my_alpha)
#axs[9,0].set_title('b l ctct-x') 
#axs[9,1].set_title('b l ctct-y')
#axs[9,2].set_title('b l ctct-z')


axs[4,0].plot(a_t, a_ctct[:,3], color='red', lw=my_lw, alpha=my_alpha)
axs[4,1].plot(a_t, a_ctct[:,4], color='red', lw=my_lw, alpha=my_alpha)
axs[4,2].plot(a_t, a_ctct[:,5], color='red', lw=my_lw, alpha=my_alpha)
axs[4,0].set_title('a r torque ctct-x') 
axs[4,1].set_title('a r torque ctct-y')
axs[4,2].set_title('a r torque ctct-z')
axs[4,0].plot(b_t, b_ctct[:,3], lw=my_lw, alpha=my_alpha)
axs[4,1].plot(b_t, b_ctct[:,4], lw=my_lw, alpha=my_alpha)
axs[4,2].plot(b_t, b_ctct[:,5], lw=my_lw, alpha=my_alpha)
#axs[7,0].set_title('b r torque ctct-x')
#axs[7,1].set_title('b r torque ctct-y')
#axs[7,2].set_title('b r torque ctct-z')

axs[5,0].plot(a_t, a_ctct[:,9], color='red', lw=my_lw, alpha=my_alpha)
axs[5,1].plot(a_t, a_ctct[:,10], color='red', lw=my_lw, alpha=my_alpha)
axs[5,2].plot(a_t, a_ctct[:,11], color='red', lw=my_lw, alpha=my_alpha)
axs[5,0].set_title('a L torque ctct-x') 
axs[5,1].set_title('a L torque ctct-y')
axs[5,2].set_title('a L torque ctct-z')

axs[5,0].plot(b_t, b_ctct[:,9], lw=my_lw, alpha=my_alpha)
axs[5,1].plot(b_t, b_ctct[:,10], lw=my_lw, alpha=my_alpha)
axs[5,2].plot(b_t, b_ctct[:,11], lw=my_lw, alpha=my_alpha)
#axs[11,0].set_title('b l torque ctct-x')
#axs[11,1].set_title('b l torque ctct-y')
#axs[11,2].set_title('b l torque ctct-z')


plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.show()



