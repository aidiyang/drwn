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


es = util.get_est_data(f)
t = es['time']
hw = util.get_real_data(r, len(t))


ss = util.snsr_breakout(es['est_snsr'])

r_t=hw['time']
print t
print r_t
fig, axs = plt.subplots(7, 3, sharex=False)

my_ls = '--'
my_lw = 2
my_alpha = 0.5
axs[0,0].plot(t, es['est_qpos'], color='b', alpha=my_alpha)
axs[0,0].plot(r_t,   hw['qpos'], color='r', alpha=my_alpha)
axs[0,0].set_title('pos')

axs[0,1].plot(t, es['est_qvel'], color='b', alpha=my_alpha)
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
plt.show()
