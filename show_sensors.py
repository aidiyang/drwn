import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt

from numpy.linalg import norm

if len(sys.argv) < 2:
    print "python [data.csv]"
    sys.exit

f = sys.argv[1]

df = pd.read_csv(f, sep=',')

t = df['time']
ctrl = df.filter(regex='ctrl').values
snsr = df.filter(regex='snsr').values
print snsr.shape

nu = ctrl.shape[1]
ns = snsr.shape[1]

qpos = snsr[:,0:20]
qvel = snsr[:,20:40]
accl = snsr[:,40:43]
gyro = snsr[:,43:46]
ctct = snsr[:,46:58]
mrkr = snsr[:,58:]
if (mrkr.shape[1] > 107):
    mrkr = mrkr.reshape(len(t), 16,4)
    print mrkr[0,:,3]
else:
    mrkr = mrkr.reshape(len(t), 16,3)
ps = np.copy(mrkr[:,:,0:3])

############### variance 
v_qpos = np.var(qpos, axis=0)
v_qvel = np.var(qvel, axis=0)
v_accl = np.var(accl, axis=0)
v_gyro = np.var(gyro, axis=0)
v_ctct = np.var(ctct, axis=0)
v_mrkr = np.var(mrkr, axis=0)
v_mrkr = np.var(mrkr, axis=0)

print "Variances:"
print "qpos:\n", v_qpos
print "qvel:\n", v_qvel
print "accl:\n", v_accl
print "gyro:\n", v_gyro
print "ctct:\n", v_ctct
print "mrkr:\n", v_mrkr

dt = np.ediff1d(t)
mean_dt = np.mean(dt)
print "Mean dt:", mean_dt

#sp_qpos = np.fft.fft(qpos, axis=0)
#sp_qvel = np.fft.fft(qvel, axis=0)
sp_accl = np.fft.fft(accl, axis=0)
f_accl = np.fft.fftfreq(len(t), d=mean_dt)

sp_gyro = np.fft.fft(gyro, axis=0)
f_gyro = np.fft.fftfreq(len(t), d=mean_dt)
sp_ctct = np.fft.fft(ctct, axis=0)
f_ctct = np.fft.fftfreq(len(t), d=mean_dt)
#sp_mrkr = np.fft.fft(mrkr, axis=0)
#sp_mrkr = np.fft.fft(mrkr, axis=0)

#f_qpos = np.fft.fftfreq(len(t), d=mean_dt)
#f_qvel = np.fft.fftfreq(len(t), d=mean_dt)
#f_mrkr = np.fft.fftfreq(len(t), d=mean_dt)
#f_mrkr = np.fft.fftfreq(len(t), d=mean_dt)


############### plot

fig, axs = plt.subplots(8, 3, sharex=False)

my_ls = '--'
my_lw = 2
my_alpha = 0.9
axs[0,0].plot(t, qpos, lw=my_lw, alpha=my_alpha)
axs[0,0].set_title('jointpos')
#axs[0,0].set_ylim([-1, 1])

axs[0,1].plot(t, qvel, lw=my_lw, alpha=my_alpha)
axs[0,1].set_title('jointvel')
#axs[0,1].set_ylim([-1, 1])

axs[0,2].plot(t, accl, lw=my_lw, alpha=my_alpha)
axs[0,2].set_title('accel')
#axs[0,2].set_ylim([-1, 1])

axs[1,0].plot(t, gyro, lw=my_lw, alpha=my_alpha)
axs[1,0].set_title('gyro')
#axs[1,0].set_ylim([-1, 1])

axs[1,1].plot(t, ctct, lw=my_lw, alpha=my_alpha)
axs[1,1].set_title('contact')
#axs[1,1].set_ylim([-1, 1])

axs[1,2].plot(t, ps[:,:,2], lw=my_lw, alpha=my_alpha)
axs[1,2].set_title('raw z-axis')
axs[1,2].set_ylim([-1, 1])

axs[2,0].plot(t, accl[:,0], lw=my_lw, alpha=my_alpha)
axs[2,0].set_title('accel-x')
axs[2,1].plot(t, accl[:,1], lw=my_lw, alpha=my_alpha)
axs[2,1].set_title('accel-y')
axs[2,2].plot(t, accl[:,2], lw=my_lw, alpha=my_alpha)
axs[2,2].set_title('accel-z')

axs[3,0].plot(f_accl, sp_accl[:,0].imag, lw=my_lw, alpha=my_alpha)
axs[3,0].set_ylim([-10, 10])
axs[3,0].set_title('fft accel-x')
axs[3,1].plot(f_accl, sp_accl[:,1].imag, lw=my_lw, alpha=my_alpha)
axs[3,1].set_ylim([-10, 10])
axs[3,1].set_title('fft accel-y')
axs[3,2].plot(f_accl, sp_accl[:,2].imag, lw=my_lw, alpha=my_alpha)
axs[3,2].set_ylim([-10, 10])
axs[3,2].set_title('fft accel-z')

axs[4,0].plot(t, gyro[:,0], color='red', lw=my_lw, alpha=my_alpha)
axs[4,0].set_title('gyro-x') 
axs[4,1].plot(t, gyro[:,1], color='red', lw=my_lw, alpha=my_alpha)
axs[4,1].set_title('gyro-y')
axs[4,2].plot(t, gyro[:,2], color='red', lw=my_lw, alpha=my_alpha)
axs[4,2].set_title('gyro-z')

axs[5,0].plot(t, ctct[:,0], color='red', lw=my_lw, alpha=my_alpha)
axs[5,0].set_title('ctct-x') 
axs[5,1].plot(t, ctct[:,1], color='red', lw=my_lw, alpha=my_alpha)
axs[5,1].set_title('ctct-y')
axs[5,2].plot(t, ctct[:,2], color='red', lw=my_lw, alpha=my_alpha)
axs[5,2].set_title('ctct-z')

axs[6,0].plot(t, ctct[:,3], color='red', lw=my_lw, alpha=my_alpha)
axs[6,0].set_title('t ctct-x') 
axs[6,1].plot(t, ctct[:,4], color='red', lw=my_lw, alpha=my_alpha)
axs[6,1].set_title('t ctct-y')
axs[6,2].plot(t, ctct[:,5], color='red', lw=my_lw, alpha=my_alpha)
axs[6,2].set_title('t ctct-z')

axs[7,0].plot(f_ctct, sp_ctct[:,3].imag, lw=my_lw, alpha=my_alpha)
axs[7,0].set_title('fft ctct-x')
axs[7,1].plot(f_ctct, sp_ctct[:,4].imag, lw=my_lw, alpha=my_alpha)
axs[7,1].set_title('fft ctct-y')
axs[7,2].plot(f_ctct, sp_ctct[:,5].imag, lw=my_lw, alpha=my_alpha)
axs[7,2].set_title('fft ctct-z')

plt.show()


