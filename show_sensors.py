import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt

from numpy.linalg import norm

import py_utils as util

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
print "std  dt:", np.std(dt)

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
start_t = 2.25 # in seconds
new_c = util.clean_mrkr_data(t, ps, conf, 3, 0.01, start_t)

min_t=np.argmax(t[:]>start_t)

print "start t:", start_t, " start index:", min_t, t[min_t]

accl = np.copy(snsr[:,40:43])
saccl = low_pass(accl, alpha);
#snsr[:,40:43] = np.copy(saccl)
gyro = np.copy(snsr[:,43:46])
sgyro = low_pass(gyro, alpha) 
#snsr[:,43:46] = np.copy(sgyro)

ctct = np.copy(snsr[:,46:58])
#sctct = low_pass(ctct, alpha) 
sctct=util.clean_ctct_data(ctct, 20)
#snsr[:,46:58] = np.copy(sctct)

#print "Initial Contact values:"
i_ctct = np.mean(ctct[0:20, :], axis=0)

#m_ctct = np.mean(ctct[:, :], axis=0)
#print "r ctct:", i_ctct[0:3]
#print "l ctct:", i_ctct[6:9]
#print "zmean ctct:", m_ctct[2] + m_ctct[8]
#print "initz ctct:", i_ctct[2] + i_ctct[8]

#new_data = np.hstack((t[:,None], ctrl, snsr, conf))
if f.startswith('clean_') == False:
    new_t = np.copy(t[min_t:]) - t[min_t]
    new_data = np.hstack((new_t[:,None], ctrl[min_t:,:], snsr[min_t:,:], new_c[min_t:,:]))
    head = 'time,'+'ctrl,'*nu+'snsr,'*ns+'conf,'*16
    np.savetxt('clean_'+f, new_data, delimiter=',', header=head, fmt='%.8f',
            comments='')

print "First datas, 0-20:"
end=20
print "accel:", accl[0,:],   "\n mean:", np.mean(accl[0:end,:], axis=0)
print "gyro :", gyro[0,:],   "\n mean:", np.mean(gyro[0:end,:], axis=0)
print "r_frc:", ctct[0,0:3], "\n mean:", np.mean(ctct[0:end,0:3], axis=0)
print "r_trq:", ctct[0,3:6], "\n mean:", np.mean(ctct[0:end,3:6], axis=0)
print "l_frc:", ctct[0,6:9], "\n mean:", np.mean(ctct[0:end,6:9], axis=0)
print "l_trq:", ctct[0,9:12],"\n mean:", np.mean(ctct[0:end,9:12], axis=0)
print "\n"


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
print "Accel Max X:", util.max_min(accl[:,0]) 
print "Accel Max Y:", util.max_min(accl[:,1]) 
print "Accel Max Z:", util.max_min(accl[:,2]) 

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
print "Gyro Max X:", util.max_min(gyro[:,0])
print "Gyro Max Y:", util.max_min(gyro[:,1])
print "Gyro Max Z:", util.max_min(gyro[:,2])

my_alpha = 0.3
axs[5,0].plot(t, ctct[:,0], color='red', lw=my_lw, alpha=my_alpha)
axs[5,1].plot(t, ctct[:,1], color='red', lw=my_lw, alpha=my_alpha)
axs[5,2].plot(t, ctct[:,2], color='red', lw=my_lw, alpha=my_alpha)
axs[5,0].set_title('ctct-x') 
axs[5,1].set_title('ctct-y')
axs[5,2].set_title('ctct-z')
axs[5,0].plot(t, sctct[:,0], color='red', ls='-.', alpha=1.0)
axs[5,1].plot(t, sctct[:,1], color='red', ls='-.', alpha=1.0)
axs[5,2].plot(t, sctct[:,2], color='red', ls='-.', alpha=1.0)
print "R Force Max/Min X:", util.max_min(ctct[:,0])
print "R Force Max/Min Y:", util.max_min(ctct[:,1])
print "R Force Max/Min Z:", util.max_min(ctct[:,2])

axs[5,0].plot(t, ctct[:,6], color='blue', lw=my_lw, alpha=my_alpha)
axs[5,1].plot(t, ctct[:,7], color='blue', lw=my_lw, alpha=my_alpha)
axs[5,2].plot(t, ctct[:,8], color='blue', lw=my_lw, alpha=my_alpha)
axs[5,0].set_title('ctct-x') 
axs[5,1].set_title('ctct-y')
axs[5,2].set_title('ctct-z')
axs[5,0].plot(t, sctct[:,6], color='blue', ls='--', alpha=1.0)
axs[5,1].plot(t, sctct[:,7], color='blue', ls='--', alpha=1.0)
axs[5,2].plot(t, sctct[:,8], color='blue', ls='--', alpha=1.0)
print "L Force Max/Min X:", util.max_min(ctct[:,6])
print "L Force Max/Min Y:", util.max_min(ctct[:,7])
print "L Force Max/Min Z:", util.max_min(ctct[:,8])

axs[6,0].plot(t, ctct[:,3], color='red', lw=my_lw, alpha=my_alpha)
axs[6,1].plot(t, ctct[:,4], color='red', lw=my_lw, alpha=my_alpha)
axs[6,2].plot(t, ctct[:,5], color='red', lw=my_lw, alpha=my_alpha)
axs[6,0].set_title('t ctct-x') 
axs[6,1].set_title('t ctct-y')
axs[6,2].set_title('t ctct-z')
axs[6,0].plot(t, sctct[:,3], color='red', ls='-.', alpha=1.0)
axs[6,1].plot(t, sctct[:,4], color='red', ls='-.', alpha=1.0)
axs[6,2].plot(t, sctct[:,5], color='red', ls='-.', alpha=1.0)
print "R Torque Max/Min X:", util.max_min(ctct[:,3])
print "R Torque Max/Min Y:", util.max_min(ctct[:,4])
print "R Torque Max/Min Z:", util.max_min(ctct[:,5])

axs[6,0].plot(t, ctct[:,9],  color='blue', lw=my_lw, alpha=my_alpha)
axs[6,1].plot(t, ctct[:,10], color='blue', lw=my_lw, alpha=my_alpha)
axs[6,2].plot(t, ctct[:,11], color='blue', lw=my_lw, alpha=my_alpha)
axs[6,0].set_title('t ctct-x') 
axs[6,1].set_title('t ctct-y')
axs[6,2].set_title('t ctct-z')
axs[6,0].plot(t, sctct[:,9],  color='blue', ls='--', alpha=1.0)
axs[6,1].plot(t, sctct[:,10], color='blue', ls='--', alpha=1.0)
axs[6,2].plot(t, sctct[:,11], color='blue', ls='--', alpha=1.0)
print "L Torque Max/Min X:", util.max_min(ctct[:,9])
print "L Torque Max/Min Y:", util.max_min(ctct[:,10])
print "L Torque Max/Min Z:", util.max_min(ctct[:,11])


axs[7,0].plot(f_ctct, sp_ctct[:,3].imag, lw=my_lw, alpha=my_alpha)
axs[7,1].plot(f_ctct, sp_ctct[:,4].imag, lw=my_lw, alpha=my_alpha)
axs[7,2].plot(f_ctct, sp_ctct[:,5].imag, lw=my_lw, alpha=my_alpha)
axs[7,0].set_title('fft ctct-x')
axs[7,1].set_title('fft ctct-y')
axs[7,2].set_title('fft ctct-z')

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.show()

#Initial Contact values: straight_walk.csv
#[  3.60828500e+00   2.38952850e+00  -1.55573900e+01   1.00925960e-02
#   3.30845850e-01   1.77191550e-01   2.52803600e+00  -2.66430235e-01
#  -1.20418400e+01   1.25703350e-01  -1.09373215e-01   5.76890150e-02]

#Initial Contact values: walk3.csv
#[  3.18307050e+00   2.65368450e+00  -1.55325300e+01   8.52495275e-03
#   3.11757550e-01   2.21288650e-01   1.98254450e+00   3.44876580e-02
#  -1.18669600e+01   5.48431650e-02  -2.21487450e-01   8.67252700e-02]

#Initial Contact values: no_walk3.csv
#[  3.56836600e+00   2.78229250e+00  -1.63210950e+01  -3.72314050e-02
#   2.99355000e-01   1.47520950e-01   2.89638250e+00   2.51768900e-01
#  -1.20343500e+01   1.18159145e-02   4.38051300e-02   1.03155000e-01]
