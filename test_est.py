import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt

import py_utils as util

f = 'out.csv'
r = 'raw.csv'
if len(sys.argv) > 1:
    f = sys.argv[1]

if len(sys.argv) > 2:
    r = sys.argv[2]


print "Getting Estimation data from", f
print "Getting Raw sensor data from", r
es = util.get_est_data(f)

#print es['time']
#t = es['time']
#print t
df = pd.read_csv(f, sep=',')
t = df['time']
#t = t - t[0]

hw = util.get_real_data(r, len(t))

ss = util.snsr_breakout(es['est_snsr'])

r_t=hw['time']
t = r_t

fig, axs = plt.subplots(7, 3, sharex=False)
print t.shape
print es['est_qpos'].shape
print hw['qpos'].shape

my_ls = '--'
my_lw = 2
my_alpha = 0.5
hw_alpha = 0.9
#axs[0,0].plot(t, es['est_qpos'], color='b', alpha=my_alpha)
axs[0,0].plot(t,   ss['qpos'], color='b', alpha=my_alpha)
axs[0,0].plot(r_t, hw['qpos'], color='r', alpha=my_alpha)
axs[0,0].set_title('pos')

#axs[0,1].plot(t, es['est_qvel'], color='b', alpha=my_alpha)
axs[0,1].plot(t, ss['qvel'], color='b', alpha=my_alpha)
axs[0,1].plot(r_t,   hw['qvel'], color='r', alpha=my_alpha)
axs[0,1].set_title('vel')
axs[0,1].set_ylim([-3, 3])

axs[0,2].plot(t, es['est_ctrl'], color='b', alpha=my_alpha)
axs[0,2].plot(r_t,   hw['ctrl'], color='r', alpha=my_alpha)
axs[0,2].set_title('ctrl')
axs[0,2].set_ylim([-1, 1])


axs[1,0].plot(t,     ss['accl'][:,0], color='b', alpha=my_alpha)
axs[1,0].plot(r_t,   hw['accl'][:,0], color='r', alpha=my_alpha)
axs[1,0].set_title('accel-x')

axs[1,1].plot(t,     ss['accl'][:,1], color='b', alpha=my_alpha)
axs[1,1].plot(r_t,   hw['accl'][:,1], color='r', alpha=my_alpha)
axs[1,1].set_title('accel-y')

axs[1,2].plot(t,     ss['accl'][:,2], color='b', alpha=my_alpha)
axs[1,2].plot(r_t,   hw['accl'][:,2], color='r', alpha=my_alpha)
axs[1,2].set_title('accel-z')

axs[2,0].plot(t,     ss['gyro'][:,0], color='b', alpha=my_alpha)
axs[2,0].plot(r_t,   hw['gyro'][:,0], color='r', alpha=my_alpha)
axs[2,0].set_title('gyro-x')

axs[2,1].plot(t,     ss['gyro'][:,1], color='b', alpha=my_alpha)
axs[2,1].plot(r_t,   hw['gyro'][:,1], color='r', alpha=my_alpha)
axs[2,1].set_title('gyro-y')

axs[2,2].plot(t,     ss['gyro'][:,2], color='b', alpha=my_alpha)
axs[2,2].plot(r_t,   hw['gyro'][:,2], color='r', alpha=my_alpha)
axs[2,2].set_title('gyro-z')

axs[3,0].plot(t,     ss['ctct'][:,0], color='b', alpha=my_alpha)
axs[3,0].plot(r_t,   hw['ctct'][:,0], color='r', alpha=my_alpha)
axs[3,1].plot(t,     ss['ctct'][:,1], color='b', alpha=my_alpha)
axs[3,1].plot(r_t,   hw['ctct'][:,1], color='r', alpha=my_alpha)
axs[3,2].plot(t,     ss['ctct'][:,2], color='b', alpha=my_alpha)
axs[3,2].plot(r_t,   hw['ctct'][:,2], color='r', alpha=my_alpha)
axs[3,0].set_title('r ctct-x')
axs[3,1].set_title('r ctct-y')
axs[3,2].set_title('r ctct-z')

axs[4,0].plot(t,     ss['ctct'][:,3], color='b', alpha=my_alpha)
axs[4,0].plot(r_t,   hw['ctct'][:,3], color='r', alpha=my_alpha)
axs[4,1].plot(t,     ss['ctct'][:,4], color='b', alpha=my_alpha)
axs[4,1].plot(r_t,   hw['ctct'][:,4], color='r', alpha=my_alpha)
axs[4,2].plot(t,     ss['ctct'][:,5], color='b', alpha=my_alpha)
axs[4,2].plot(r_t,   hw['ctct'][:,5], color='r', alpha=my_alpha)
axs[4,0].set_title('r t ctct-x')
axs[4,1].set_title('r t ctct-y')
axs[4,2].set_title('r t ctct-x')

axs[5,0].plot(t,     ss['ctct'][:,6+0], color='b', alpha=my_alpha)
axs[5,0].plot(r_t,   hw['ctct'][:,6+0], color='r', alpha=my_alpha)
axs[5,1].plot(t,     ss['ctct'][:,6+1], color='b', alpha=my_alpha)
axs[5,1].plot(r_t,   hw['ctct'][:,6+1], color='r', alpha=my_alpha)
axs[5,2].plot(t,     ss['ctct'][:,6+2], color='b', alpha=my_alpha)
axs[5,2].plot(r_t,   hw['ctct'][:,6+2], color='r', alpha=my_alpha)
axs[5,0].set_title('l ctct-x')
axs[5,1].set_title('l ctct-y')
axs[5,2].set_title('l ctct-z')

axs[6,0].plot(t,     ss['ctct'][:,6+3], color='b', alpha=my_alpha)
axs[6,0].plot(r_t,   hw['ctct'][:,6+3], color='r', alpha=my_alpha)
axs[6,1].plot(t,     ss['ctct'][:,6+4], color='b', alpha=my_alpha)
axs[6,1].plot(r_t,   hw['ctct'][:,6+4], color='r', alpha=my_alpha)
axs[6,2].plot(t,     ss['ctct'][:,6+5], color='b', alpha=my_alpha)
axs[6,2].plot(r_t,   hw['ctct'][:,6+5], color='r', alpha=my_alpha)
axs[6,0].set_title('l t ctct-x')
axs[6,1].set_title('l t ctct-y')
axs[6,2].set_title('l t ctct-z')

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

############ marker distance vs estimated difference

#fig3, axs3 = plt.subplots(1, 1, sharex=False)
#
#rms = util.dist_diff_v_time(ss['ps'], hw['ps'], hw['conf'])
#
#print "rms shape:", rms.shape
#
#axs3.plot(t, rms, color='b', alpha=my_alpha)
##axs3[0,0].plot(r_t, hw['ps'][:,i,0], color='r', alpha=hw_alpha)
#
#plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

############ marker plots


# fig2, axs2 = plt.subplots(16, 3, sharex=False)
# s_x = ss['ps'][:,:,0]
# s_y = ss['ps'][:,:,1]
# s_z = ss['ps'][:,:,2]
# 
# r_x = hw['ps'][:,:,0]
# r_y = hw['ps'][:,:,1]
# r_z = hw['ps'][:,:,2]
# 
# #print "Sim Mean x:\n", np.mean(s_x, axis=0);
# #print "Sim Mean y:\n", np.mean(s_y, axis=0);
# #print "Sim Mean z:\n", np.mean(s_z, axis=0);
# #print "IRL Mean x:\n", np.mean(r_x, axis=0);
# #print "IRL Mean y:\n", np.mean(r_y, axis=0);
# #print "IRL Mean z:\n", np.mean(r_z, axis=0);
# 
# #print "Sim Var x:\n", np.var(s_x, axis=0);
# #print "Sim Var y:\n", np.var(s_y, axis=0);
# #print "Sim Var z:\n", np.var(s_z, axis=0);
# #print "IRL Var x:\n", np.var(r_x, axis=0);
# #print "IRL Var y:\n", np.var(r_y, axis=0);
# #print "IRL Var z:\n", np.var(r_z, axis=0);
# 
# for i in range(0,16):
#     axs2[i,0].plot(t,   ss['ps'][:,i,0], color='b', alpha=my_alpha)
#     axs2[i,0].plot(r_t, hw['ps'][:,i,0], color='r', alpha=hw_alpha)
#     axs2[i,1].plot(t,   ss['ps'][:,i,1], color='b', alpha=my_alpha)
#     axs2[i,1].plot(r_t, hw['ps'][:,i,1], color='r', alpha=hw_alpha)
#     axs2[i,2].plot(t,   ss['ps'][:,i,2], color='b', alpha=my_alpha)
#     axs2[i,2].plot(r_t, hw['ps'][:,i,2], color='r', alpha=hw_alpha)
# axs2[0,0].set_title('mrkr-x')
# axs2[0,1].set_title('mrkr-y')
# axs2[0,2].set_title('mrkr-z')
# 
# plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)



plt.show()
