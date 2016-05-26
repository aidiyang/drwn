import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt

assert len(sys.argv) > 1

f = sys.argv[1]
df = pd.read_csv(f, sep=',')

t = df['time']

qpos = df.filter(regex='qpos').values
qvel = df.filter(regex='qvel').values
ctrl = df.filter(regex='ctrl').values
snsr = df.filter(regex='snsr').values

est_qpos = df.filter(regex='est_p').values
est_qvel = df.filter(regex='est_v').values
est_ctrl = df.filter(regex='est_c').values
est_snsr = df.filter(regex='est_s').values

nq = est_qpos.shape[1]
nv = est_qvel.shape[1]
nu = est_ctrl.shape[1]
ns = est_snsr.shape[1]

std_qpos = df.filter(regex='stddev_p').values
std_qvel = df.filter(regex='stddev_v').values
std_ctrl = df.filter(regex='stddev_c').values
std_snsr = df.filter(regex='stddev_s').values
std_qpos = np.sqrt(std_qpos) 
std_qvel = np.sqrt(std_qvel) 
std_ctrl = np.sqrt(std_ctrl) 
std_snsr = np.sqrt(std_snsr) 

p_time = df['predict']
c_time = df['correct']


fig, axs = plt.subplots(2, 3, sharex=False)

my_ls = '--'
my_lw = 5
my_alpha = 0.1
if qpos.any():
    axs[0,0].plot(t, qpos, lw=my_lw, alpha=my_alpha)
plt.gca().set_color_cycle(None) # reset color cycle
axs[0,0].plot(t, est_qpos, ls=my_ls, alpha=1.0)
#axs[1,0].fill_between(t, est_qpos+std_qpos, est_qpos-std_qpos, ls=my_ls, alpha=1.0)
axs[0,0].set_title('qpos')
for col in range(nq):
    axs[0,0].fill_between(t, est_qpos[:,col]+std_qpos[:,col],
            est_qpos[:,col]-std_qpos[:,col], edgecolor='none', alpha=0.1)

if qvel.any():
    axs[0,1].plot(t, qvel, lw=my_lw, alpha=my_alpha)
plt.gca().set_color_cycle(None)
axs[0,1].plot(t, est_qvel, ls=my_ls, alpha=1.0)
#axs[1,0].fill_between(t, est_qvel+std_qvel, est_qvel-std_qvel, ls=my_ls, alpha=1.0)
axs[0,1].set_title('qvel')
for col in range(nv):
    axs[0,1].fill_between(t, est_qvel[:,col]+std_qvel[:,col],
            est_qvel[:,col]-std_qvel[:,col], edgecolor='none', alpha=0.1)

if ctrl.any():
    axs[1,0].plot(t, ctrl, lw=my_lw, alpha=my_alpha)
plt.gca().set_color_cycle(None)
axs[1,0].plot(t, est_ctrl, ls=my_ls, alpha=1.0)
#axs[1,0].fill_between(t, est_ctrl+std_ctrl, est_ctrl-std_ctrl, ls=my_ls, alpha=1.0)
axs[1,0].set_title('ctrl')
#for col in range(nu):
#    axs[1,0].fill_between(t, est_ctrl[:,col]+std_ctrl[:,col],
#            est_ctrl[:,col]-std_ctrl[:,col], edgecolor='none', alpha=0.1)

if snsr.any():
    axs[1,1].plot(t, snsr, lw=my_lw, alpha=my_alpha)
plt.gca().set_color_cycle(None)
axs[1,1].plot(t, est_snsr, ls=my_ls, alpha=1.0)
#axs[1,0].fill_between(t, est_snsr+std_snsr, est_snsr-std_snsr, ls=my_ls, alpha=1.0)
axs[1,1].set_title('sensors')
for col in range(ns):
    axs[1,1].fill_between(t, est_snsr[:,col]+std_snsr[:,col],
            est_snsr[:,col]-std_snsr[:,col], edgecolor='none', alpha=0.1)

axs[0,2].plot(t, p_time, ls=my_ls, alpha=1.0, label="Prediction")
axs[0,2].set_title('timings')
axs[0,2].plot(t, c_time, ls=my_ls, alpha=1.0, label="Correction")
axs[0,2].plot(t, c_time+p_time, ls=my_ls, alpha=1.0, label="Total")
axs[0,2].legend()

print "Prediction Timing Avg:", np.mean(p_time)
print "Correction Timing Avg:", np.mean(c_time)
print "     Total Timing Avg:", np.mean(c_time+p_time)

#axs[0,2].set_title('correct time')

#axs[1,2].legend()
#for col in range(nq):
#    axs[1,2].fill_between(t, est_qpos[:,col]+std_qpos[:,col],
#            est_qpos[:,col]-std_qpos[:,col], edgecolor='none', alpha=0.5)

plt.show()
