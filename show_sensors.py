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
conf = df.filter(regex='conf').values
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


print mrkr[0,:,:], conf[0,:]

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
print "Var  dt:", np.std(dt)

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

def moving_avg(a, n=5):
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n

#ctct = pd.rolling_mean(ctct, 20)

for i in range(0,16):
    m = np.mean(conf[:,i])
    print m
    if m < 5:
        conf[:,i] = -1


def low_pass(raw, alpha):
    p = np.mean(raw, axis=0)
    filt = np.zeros(raw.shape)
    for i in range(0, len(t)):
        filt[i,:] = p + alpha * (raw[i,:]-p)
        p=filt[i,:] 
    return filt

print "Accl mean:", np.mean(accl, axis=0)
print "Gyro mean:", np.mean(gyro, axis=0)

alpha = 0.5
accl = np.copy(snsr[:,40:43])
saccl = low_pass(accl, alpha);
snsr[:,40:43] = np.copy(saccl)

gyro = np.copy(snsr[:,43:46])
sgyro = low_pass(gyro, alpha) 
snsr[:,43:46] = np.copy(sgyro)

ctct = np.copy(snsr[:,46:58])
sctct = low_pass(ctct, alpha) 
snsr[:,46:58] = np.copy(sctct)


new_data = np.hstack((t[:,None], ctrl, snsr, conf))
head = 'time,'+'ctrl,'*nu+'snsr,'*ns+'conf,'*16
np.savetxt('clean_'+f, new_data, delimiter=',', header=head, fmt='%.8f',
        comments='')

############### plot

fig, axs = plt.subplots(8, 3, sharex=False)

my_ls = '--'
my_lw = 2
my_alpha = 0.5
axs[0,0].plot(t, qpos, lw=my_lw, alpha=my_alpha)
axs[0,0].set_title('jointpos')
#axs[0,0].set_ylim([-1, 1])

axs[0,1].plot(t, qvel, lw=my_lw, alpha=my_alpha)
axs[0,1].set_title('jointvel')
#axs[0,1].set_ylim([-1, 1])

axs[0,2].plot(t, ctrl, lw=my_lw, alpha=my_alpha)
axs[0,2].set_title('ctrl')
#axs[0,2].set_ylim([-1, 1])

axs[1,0].plot(t, ps[:,:,0], lw=my_lw, alpha=my_alpha)
axs[1,0].set_title('raw x-axis')
#axs[1,0].set_ylim([-1, 1])

axs[1,1].plot(t, ps[:,:,1], lw=my_lw, alpha=my_alpha)
axs[1,1].set_title('raw y-axis')
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
axs[2,0].plot(t, saccl[:,0], color='red', lw=my_lw, alpha=my_alpha)
axs[2,1].plot(t, saccl[:,1], color='red', lw=my_lw, alpha=my_alpha)
axs[2,2].plot(t, saccl[:,2], color='red', lw=my_lw, alpha=my_alpha)

axs[3,0].plot(f_accl, sp_accl[:,0].imag, lw=my_lw, alpha=my_alpha)
axs[3,0].set_ylim([-10, 10])
axs[3,0].set_title('fft accel-x')
axs[3,1].plot(f_accl, sp_accl[:,1].imag, lw=my_lw, alpha=my_alpha)
axs[3,1].set_ylim([-10, 10])
axs[3,1].set_title('fft accel-y')
axs[3,2].plot(f_accl, sp_accl[:,2].imag, lw=my_lw, alpha=my_alpha)
axs[3,2].set_ylim([-10, 10])
axs[3,2].set_title('fft accel-z')

axs[4,0].plot(t, gyro[:,0], color='blue', lw=my_lw, alpha=my_alpha)
axs[4,0].set_title('gyro-x') 
axs[4,1].plot(t, gyro[:,1], color='blue', lw=my_lw, alpha=my_alpha)
axs[4,1].set_title('gyro-y')
axs[4,2].plot(t, gyro[:,2], color='blue', lw=my_lw, alpha=my_alpha)
axs[4,2].set_title('gyro-z')
axs[4,0].plot(t, sgyro[:,0], color='red', lw=my_lw, alpha=my_alpha)
axs[4,1].plot(t, sgyro[:,1], color='red', lw=my_lw, alpha=my_alpha)
axs[4,2].plot(t, sgyro[:,2], color='red', lw=my_lw, alpha=my_alpha)

my_alpha = 0.3
axs[5,0].plot(t, ctct[:,0], color='red', lw=my_lw, alpha=my_alpha)
axs[5,1].plot(t, ctct[:,1], color='red', lw=my_lw, alpha=my_alpha)
axs[5,2].plot(t, ctct[:,2], color='red', lw=my_lw, alpha=my_alpha)
axs[5,0].set_title('ctct-x') 
axs[5,1].set_title('ctct-y')
axs[5,2].set_title('ctct-z')
axs[5,0].plot(t, sctct[:,0], color='red', ls='--', alpha=1.0)
axs[5,1].plot(t, sctct[:,1], color='red', ls='--', alpha=1.0)
axs[5,2].plot(t, sctct[:,2], color='red', ls='--', alpha=1.0)

axs[5,0].plot(t, ctct[:,6], color='blue', lw=my_lw, alpha=my_alpha)
axs[5,1].plot(t, ctct[:,7], color='blue', lw=my_lw, alpha=my_alpha)
axs[5,2].plot(t, ctct[:,8], color='blue', lw=my_lw, alpha=my_alpha)
axs[5,0].set_title('ctct-x') 
axs[5,1].set_title('ctct-y')
axs[5,2].set_title('ctct-z')
axs[5,0].plot(t, sctct[:,6], color='blue', ls='--', alpha=1.0)
axs[5,1].plot(t, sctct[:,7], color='blue', ls='--', alpha=1.0)
axs[5,2].plot(t, sctct[:,8], color='blue', ls='--', alpha=1.0)


axs[6,0].plot(t, ctct[:,3], color='red', lw=my_lw, alpha=my_alpha)
axs[6,1].plot(t, ctct[:,4], color='red', lw=my_lw, alpha=my_alpha)
axs[6,2].plot(t, ctct[:,5], color='red', lw=my_lw, alpha=my_alpha)
axs[6,0].set_title('t ctct-x') 
axs[6,1].set_title('t ctct-y')
axs[6,2].set_title('t ctct-z')
axs[6,0].plot(t, sctct[:,3], color='red', ls='--', alpha=1.0)
axs[6,1].plot(t, sctct[:,4], color='red', ls='--', alpha=1.0)
axs[6,2].plot(t, sctct[:,5], color='red', ls='--', alpha=1.0)

axs[6,0].plot(t, ctct[:,9],  color='blue', lw=my_lw, alpha=my_alpha)
axs[6,1].plot(t, ctct[:,10], color='blue', lw=my_lw, alpha=my_alpha)
axs[6,2].plot(t, ctct[:,11], color='blue', lw=my_lw, alpha=my_alpha)
axs[6,0].set_title('t ctct-x') 
axs[6,1].set_title('t ctct-y')
axs[6,2].set_title('t ctct-z')
axs[6,0].plot(t, sctct[:,9],  color='blue', ls='--', alpha=1.0)
axs[6,1].plot(t, sctct[:,10], color='blue', ls='--', alpha=1.0)
axs[6,2].plot(t, sctct[:,11], color='blue', ls='--', alpha=1.0)


axs[7,0].plot(f_ctct, sp_ctct[:,3].imag, lw=my_lw, alpha=my_alpha)
axs[7,1].plot(f_ctct, sp_ctct[:,4].imag, lw=my_lw, alpha=my_alpha)
axs[7,2].plot(f_ctct, sp_ctct[:,5].imag, lw=my_lw, alpha=my_alpha)
axs[7,0].set_title('fft ctct-x')
axs[7,1].set_title('fft ctct-y')
axs[7,2].set_title('fft ctct-z')

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.show()


