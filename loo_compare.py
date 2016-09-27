import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt

import py_utils as util


if len(sys.argv) >= 2:
    hw_f = sys.argv[1]

print "Hardware File:", hw_f
hw = util.get_real_data(hw_f, -1)

my_lw = 2
lw2 = 2
my_alpha = 0.9
conf_lvl = 0
conf = hw['conf']
my_ls = ['-','--','-.',':']

#conf[:,0] = -1
#conf[:,1] = -1

new_c = util.clean_mrkr_data(hw['time'], hw['ps'], conf, 3, 0.01, 0)

if hw_f.startswith('clean_fallen') == True:
    #cols = [1, 2] # n_fall_1
    #cols = [1, 2, 11] # n_fall_4
    cols = [2, 5, 6, 11] # n_fall_3
    init_spos = np.array([0.4273  ,0.0566  ,0.1082  ,0.3610 ,-0.0552 , 0.1390,   
       0.4205 , 0.0471 , 0.0602 , 0.3797 , 0.0206  ,0.0303  ,0.4283  ,-0.0554,
       0.1046 , 0.3599 , 0.0528 , 0.1425 , 0.4213  ,-0.0429 , 0.0572 , 0.3801,
       -0.0154,  0.0291,  0.0669,  -0.0748,  0.0980,  0.0648,  0.0706,  0.0995,
       0.0730 , -0.0725,  0.0101,  0.0725 , 0.0732 , 0.0117 , 0.3261 , -0.1067,
       0.0520 , 0.3018 , -0.1156,  0.0323 , 0.3258 , 0.1097 , 0.0504 , 0.3035,
       0.1196 , 0.0289])
    print init_spos.shape
    init_spos = init_spos.reshape(16,3)
    print "fallen cols is: ", cols

elif hw_f.startswith('clean_straight') == True:
    #cols = [2,4,6,7,8,11,12] # straight walk
    cols = [2] # straight walk full
    #init_spos = np.zeros((16,3))
    #init_spos = np.array([0.0112 , 0.0562 , 0.3778 , -0.0361,  -0.0537,  0.3186,  0.0566 , 0.0452  ,0.3616,
    #0.0768 , 0.0181 , 0.3156 , 0.0110 , -0.0558 , 0.3778 , -0.0359 , 0.0543  ,0.3186,
    #0.0564 , -0.0448,  0.3616,  0.0767,  -0.0179,  0.3156,  -0.0446,  -0.0733,  0.0175,
    #-0.0443,  0.0740,  0.0175,  0.0434,  -0.0733,  0.0129,  0.0437 , 0.0737  ,0.0129,
    #0.0401 , -0.1078,  0.2669,  0.0541,  -0.1171,  0.2392,  0.0505 , 0.1083  ,0.2672,
    #0.0679 , 0.1176 , 0.2414])
    #init_spos = init_spos.reshape(16,3)
    init_spos = hw['ps'][0,:,:]
    print "straight walk cols is: ", cols

init_delta = hw['ps'][0,:,:] - init_spos
hw['ps'] = hw['ps'][:,:,:] - init_delta

fig, axs = plt.subplots(4, 3, sharex=True, figsize=(15,10))

f0 = 'all_snsr.csv' 
f1 = 'no_accel.csv' 
f2 = 'no_gyro.csv' 
f3 = 'no_force.csv' 
f4 = 'no_torque.csv' 
f5 = 'no_jpos.csv' 
f6 = 'no_jvel.csv' 

good = util.get_est_data(f0)
fu = util.snsr_breakout(good['est_snsr'])
tg = good['time']

files = [f0,f1,f2,f3,f4,f5,f6]
names = ['All', 'No Accl', 'No Gyro', 'No Force', 'No Torque', 'No J.Pos', 'No J.Vel']

c1 = 'red'
c2 = 'blue'
c3 = 'black'


es = util.get_est_data(f1)
t = es['time']
tr = hw['time']
ss = util.snsr_breakout(es['est_snsr'])
s = ss['accl']
r = hw['accl']
g = fu['accl']
for i in range(3):
    axs[0,i].plot(t, s[:,i], color=c1, lw=my_lw, alpha=my_alpha, label=f1)
    axs[0,i].plot(tr,r[:,i], color=c2, lw=my_lw, alpha=my_alpha, label=f1)
    axs[0,i].plot(tg,g[:,i], color=c3, lw=lw2, ls='--', alpha=my_alpha, label=f1)
axs[0,1].set_title('Accelerometer')

es = util.get_est_data(f2)
t = es['time']
ss = util.snsr_breakout(es['est_snsr'])
s = ss['gyro']
r = hw['gyro']
g = fu['gyro']
for i in range(3):
    axs[1,i].plot(t, s[:,i], color=c1, lw=my_lw, alpha=my_alpha, label=f2)
    axs[1,i].plot(tr,r[:,i], color=c2, lw=my_lw, alpha=my_alpha, label=f2)
    axs[1,i].plot(tg,g[:,i], color=c3, lw=lw2, ls='--', alpha=my_alpha, label=f0)
axs[1,1].set_title('Gyroscope')

es = util.get_est_data(f3)
t = es['time']
ss = util.snsr_breakout(es['est_snsr'])
s = ss['ctct'][:,0:3]
r = hw['ctct'][:,0:3]
g = fu['ctct'][:,0:3]
print "ctct f shape:", s.shape
for i in range(3):
    axs[2,i].plot(t, s[:,i], color=c1, lw=my_lw, alpha=my_alpha, label=f3)
    axs[2,i].plot(tr,r[:,i], color=c2, lw=my_lw, alpha=my_alpha, label=f3)
    axs[2,i].plot(tg,g[:,i], color=c3, lw=lw2, ls='--', alpha=my_alpha, label=f0)
axs[2,1].set_title('Force Sensors (Right)')

es = util.get_est_data(f4)
t = es['time']
ss = util.snsr_breakout(es['est_snsr'])
s = ss['ctct'][:,3:6]
r = hw['ctct'][:,3:6]
g = fu['ctct'][:,3:6]
print "ctct t shape:", s.shape
for i in range(3):
    axs[3,i].plot(t, s[:,i], color=c1, lw=my_lw, alpha=my_alpha, label=f4)
    axs[3,i].plot(tr,r[:,i], color=c2, lw=my_lw, alpha=my_alpha, label=f4)
    axs[3,i].plot(tg,g[:,i], color=c3, lw=lw2, ls='--', alpha=my_alpha, label=f0)
axs[3,1].set_title('Torque Sensors (Right)')

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

start = 2.50
end = 4.50
plt.xlim(start, end)

fig2, axs2 = plt.subplots(1,1, sharex=True, figsize=(15,10))

for f in range(len(files)):
    r = files[f]
    es = util.get_est_data(r)
    t = es['time']
    ss = util.snsr_breakout(es['est_snsr'])
    
    #rms = util.dist_diff_v_time(ss['ps'], hw['ps'], conf, conf_lvl)
    #rms = util.dist_diff_v_time(ss['ps'], hw['ps'], new_c, conf_lvl)
    rms = util.dist_diff_v_time_limited(ss['ps'], hw['ps'], new_c, conf_lvl, cols)
    
    idx = (f-2)%len(my_ls)
    print "ls:", idx
    axs2.plot(t, rms, ls=my_ls[idx],
        lw=3, alpha=1.0, label=names[f])

axs2.set_title('RMS of Leave-One-Out')
axs2.set_xlabel('Time')
axs2.set_ylabel('RMSE (Meters)')
axs2.grid(True)
axs2.legend()


ymin, ymax = plt.ylim()
if ymax > 1:
    ymax = 0.3
    plt.ylim(ymin, ymax)

#axs.legend()
#plt.legend(bbox_to_anchor=(0.5, 0.8), bbox_transform=plt.gcf().transFigure)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

plt.show()


