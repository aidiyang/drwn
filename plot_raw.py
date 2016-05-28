import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt

f = 'raw.csv'
if len(sys.argv) > 1:
    f = sys.argv[1]

df = pd.read_csv(f, sep=',')

t = df['time']
dt = np.diff(t)
dt_ms = np.mean(dt)*1000.0;
dt_st = np.std(dt*1000.0);

print "      Timing Avg:", dt_ms, "ms "
print "   Timing stddev:", dt_st, "ms "

#qpos = df.filter(regex='qpos').values
#qvel = df.filter(regex='qvel').values
ctrl = df.filter(regex='ctrl').values
snsr = df.filter(regex='snsr').values

#est_qpos = df.filter(regex='est_p').values
#est_qvel = df.filter(regex='est_v').values
#est_ctrl = df.filter(regex='est_c').values
#est_snsr = df.filter(regex='est_s').values

#nq = est_qpos.shape[1]
#nv = est_qvel.shape[1]
nu = ctrl.shape[1]
ns = snsr.shape[1]

#std_qpos = df.filter(regex='stddev_p').values
#std_qvel = df.filter(regex='stddev_v').values
#std_ctrl = df.filter(regex='stddev_c').values
#std_snsr = df.filter(regex='stddev_s').values
#std_qpos = np.sqrt(std_qpos) 
#std_qvel = np.sqrt(std_qvel) 
#std_ctrl = np.sqrt(std_ctrl) 
#std_snsr = np.sqrt(std_snsr) 

#p_time = df['predict']
#c_time = df['correct']


fig, axs = plt.subplots(2, 1, sharex=False)

my_ls = '--'
my_lw = 5
my_alpha = 0.1

c_mean = np.mean(ctrl, axis=0)
print nu
print np.shape(c_mean)
print "snsr mean", c_mean


if ctrl.any():
    axs[0].plot(t, ctrl, lw=my_lw, alpha=my_alpha)
plt.gca().set_color_cycle(None)
#axs[1,0].plot(t, est_ctrl, ls=my_ls, alpha=1.0)
#axs[1,0].fill_between(t, est_ctrl+std_ctrl, est_ctrl-std_ctrl, ls=my_ls, alpha=1.0)
axs[0].set_title('ctrl')
#for col in range(nu):
#    axs[1,0].fill_between(t, est_ctrl[:,col]+std_ctrl[:,col],
#            est_ctrl[:,col]-std_ctrl[:,col], edgecolor='none', alpha=0.1)

s_mean = np.mean(snsr, axis=0)
s_var = np.var(snsr, axis=0)
print ns
print np.shape(s_mean)
print "snsr mean", s_mean
print "snsr var position", s_var[0:20]
print "snsr var velocity", s_var[20:40]
print "snsr var acclrtin", s_var[40:43]
print "snsr var gyroscps", s_var[43:46]
print "snsr var r frctrq", s_var[46:52]
print "snsr var l frctrq", s_var[52:58]

if snsr.any():
    axs[1].plot(t, snsr[:,0:20], lw=my_lw, alpha=my_alpha)
plt.gca().set_color_cycle(None)
#Jaxs[1,1].plot(t, est_snsr, ls=my_ls, alpha=1.0)
#axs[1,0].fill_between(t, est_snsr+std_snsr, est_snsr-std_snsr, ls=my_ls, alpha=1.0)
axs[1].set_title('sensors')
#for col in range(ns):
#    axs[1,1].fill_between(t, est_snsr[:,col]+std_snsr[:,col],
#            est_snsr[:,col]-std_snsr[:,col], edgecolor='none', alpha=0.1)

#axs[0,2].plot(t, p_time, ls=my_ls, alpha=1.0, label="Prediction")
#axs[0,2].set_title('timings')
#axs[0,2].plot(t, c_time, ls=my_ls, alpha=1.0, label="Correction")
#axs[0,2].plot(t, c_time+p_time, ls=my_ls, alpha=1.0, label="Total")
#axs[0,2].legend()

#print "Total Timing Avg:", np.mean(time+p_time)

#axs[0,2].set_title('correct time')

#axs[1,2].legend()
#for col in range(nq):
#    axs[1,2].fill_between(t, est_qpos[:,col]+std_qpos[:,col],
#            est_qpos[:,col]-std_qpos[:,col], edgecolor='none', alpha=0.5)

plt.show()

