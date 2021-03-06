import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt

f = 'out.csv'
r = 'raw.csv'
if len(sys.argv) > 1:
    f = sys.argv[1]

df = pd.read_csv(f, sep=',')

t = df['time']
print t

qpos = df.filter(regex='qpos').values
qvel = df.filter(regex='qvel').values
ctrl = df.filter(regex='ctrl').values
snsr = df.filter(regex='snsr').values
#mrkr = snsr[:,58:]
#mrkr = mrkr.reshape(len(t), 16,3)

est_qpos = df.filter(regex='est_p').values
est_qvel = df.filter(regex='est_v').values
est_ctrl = df.filter(regex='est_c').values
est_snsr = df.filter(regex='est_s').values
est_mrkr = est_snsr[:,58:(58+16*3)]
if (est_mrkr.any()):
    est_mrkr = est_mrkr.reshape(len(est_mrkr), 16,3)

nq = est_qpos.shape[1]
nv = est_qvel.shape[1]
nu = est_ctrl.shape[1]
ns = est_snsr.shape[1]

print "nq:", nq
print "nv:", nv
print "nu:", nu
print "ns:", ns

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



fig, axs = plt.subplots(4, 3, sharex=False)

my_ls = '--'
my_lw = 5
my_alpha = 0.1
root = 6
if (root > nq):
    root = nq

if qpos.any():
    axs[0,0].plot(t, qpos[:,0:root], lw=my_lw, alpha=my_alpha)
plt.gca().set_color_cycle(None) # reset color cycle
axs[0,0].plot(t, est_qpos[:,0:root], ls=my_ls, alpha=1.0)
axs[0,0].set_title('qpos root')
for col in range(0,root):
    axs[0,0].fill_between(t, est_qpos[:,col]+std_qpos[:,col],
            est_qpos[:,col]-std_qpos[:,col], edgecolor='none', alpha=0.1)

if (root == 6 ):
    if qpos.any():
        axs[1,0].plot(t, qpos[:,root:], lw=my_lw, alpha=my_alpha)
    plt.gca().set_color_cycle(None) # reset color cycle
    axs[1,0].plot(t, est_qpos[:,root:], ls=my_ls, alpha=1.0)
    axs[1,0].set_title('qpos joints')
    for col in range(root, nq):
        axs[1,0].fill_between(t, est_qpos[:,col]+std_qpos[:,col],
                est_qpos[:,col]-std_qpos[:,col], edgecolor='none', alpha=0.1)


if qvel.any():
    axs[0,1].plot(t, qvel[:,0:root], lw=my_lw, alpha=my_alpha)
plt.gca().set_color_cycle(None)
axs[0,1].plot(t, est_qvel[:,0:root], ls=my_ls, alpha=1.0)
axs[0,1].set_title('qvel root, solid=sim, dotted=est')
for col in range(0, root):
    axs[0,1].fill_between(t, est_qvel[:,col]+std_qvel[:,col],
            est_qvel[:,col]-std_qvel[:,col], edgecolor='none', alpha=0.1)


if (root == 6 ):
    if qvel.any():
        axs[1,1].plot(t, qvel[:,root:], lw=my_lw, alpha=my_alpha)
    plt.gca().set_color_cycle(None)
    axs[1,1].plot(t, est_qvel[:,root:], ls=my_ls, alpha=1.0)
    axs[1,1].set_title('qvel joints')
    for col in range(root, nv):
        axs[1,1].fill_between(t, est_qvel[:,col]+std_qvel[:,col],
                est_qvel[:,col]-std_qvel[:,col], edgecolor='none', alpha=0.1)


if ctrl.any():
    axs[2,0].plot(t, ctrl, lw=my_lw, alpha=my_alpha)
plt.gca().set_color_cycle(None)

if est_ctrl.any():
    axs[2,0].plot(t, est_ctrl, ls=my_ls, alpha=1.0)
axs[2,0].set_title('ctrl')
#for col in range(nu):
#    axs[1,0].fill_between(t, est_ctrl[:,col]+std_ctrl[:,col],
#            est_ctrl[:,col]-std_ctrl[:,col], edgecolor='none', alpha=0.1)

# accel
if (ns > 56):
    if snsr[:,40:43].any():
        axs[2,1].plot(t, snsr[:,40:43], lw=my_lw, alpha=my_alpha)
    plt.gca().set_color_cycle(None)
    axs[2,1].plot(t, est_snsr[:,40:43], ls=my_ls, alpha=1.0)
    #axs[1,0].fill_between(t, est_snsr+std_snsr, est_snsr-std_snsr, ls=my_ls, alpha=1.0)
    axs[2,1].set_title('accel')
    for col in range(40,43):
        axs[2,1].fill_between(t, est_snsr[:,col]+std_snsr[:,col],
                est_snsr[:,col]-std_snsr[:,col], edgecolor='none', alpha=0.1)

    print "Accel Max X:", np.max(est_snsr[:,40]), np.min(est_snsr[:,40])
    print "Accel Max Y:", np.max(est_snsr[:,41]), np.min(est_snsr[:,41])
    print "Accel Max Z:", np.max(est_snsr[:,42]), np.min(est_snsr[:,42])

# gyro
if (ns > 56):
    if snsr.any():
        axs[2,2].plot(t, snsr[:,43:46], lw=my_lw, alpha=my_alpha)
    plt.gca().set_color_cycle(None)
    axs[2,2].plot(t, est_snsr[:,43:46], ls=my_ls, alpha=1.0)
    #axs[1,0].fill_between(t, est_snsr+std_snsr, est_snsr-std_snsr, ls=my_ls, alpha=1.0)
    axs[2,2].set_title('gyro')
    for col in range(43,46):
        axs[2,2].fill_between(t, est_snsr[:,col]+std_snsr[:,col],
                est_snsr[:,col]-std_snsr[:,col], edgecolor='none', alpha=0.1)
    print "Gyro Max X:", np.max(est_snsr[:,43]), np.min(est_snsr[:,43])
    print "Gyro Max Y:", np.max(est_snsr[:,44]), np.min(est_snsr[:,44])
    print "Gyro Max Z:", np.max(est_snsr[:,45]), np.min(est_snsr[:,45])

## mrkr 
# if mrkr.any():
#     axs[1,2].plot(t, mrkr[:,7,:], lw=my_lw, alpha=my_alpha)
# plt.gca().set_color_cycle(None)
# axs[1,2].plot(t, est_mrkr[:,7,:], ls=my_ls, alpha=1.0)
# #axs[1,0].fill_between(t, est_snsr+std_snsr, est_snsr-std_snsr, ls=my_ls, alpha=1.0)
# axs[1,2].set_title('mrkr 7')
# #for col in range(40,43):
# #    axs[2,1].fill_between(t, est_mrkr[:,col]+std_mrkr[:,col],
# #            est_mrkr[:,col]-std_mrkrk[:,col], edgecolor='none', alpha=0.1)

# force / torque 
if (ns > 56):
    if snsr.any():
        axs[3,1].plot(t, snsr[:,46:49], lw=my_lw, color='blue', alpha=my_alpha)
        axs[3,1].plot(t, snsr[:,52:55], lw=my_lw, color='red', alpha=my_alpha)
    plt.gca().set_color_cycle(None)
    axs[3,1].plot(t, est_snsr[:,46:48], ls=my_ls, alpha=1.0)
    axs[3,1].plot(t, est_snsr[:,52:55], ls=my_ls, alpha=1.0)
    #axs[1,0].fill_between(t, est_snsr+std_snsr, est_snsr-std_snsr, ls=my_ls, alpha=1.0)
    axs[3,1].set_title('force')
    for col in range(46,48):
        axs[3,1].fill_between(t, est_snsr[:,col]+std_snsr[:,col],
                est_snsr[:,col]-std_snsr[:,col], edgecolor='none', alpha=0.1)
    for col in range(52,55):
        axs[3,1].fill_between(t, est_snsr[:,col]+std_snsr[:,col],
                est_snsr[:,col]-std_snsr[:,col], edgecolor='none', alpha=0.1)
        
    print "R Force Max X:", np.max(est_snsr[:,46]), np.min(est_snsr[:,46])
    print "R Force Max Y:", np.max(est_snsr[:,47]), np.min(est_snsr[:,47])
    print "R Force Max Z:", np.max(est_snsr[:,48]), np.min(est_snsr[:,48])
    print "L Force Max X:", np.max(est_snsr[:,52]), np.min(est_snsr[:,52])
    print "L Force Max Y:", np.max(est_snsr[:,53]), np.min(est_snsr[:,53])
    print "L Force Max Z:", np.max(est_snsr[:,54]), np.min(est_snsr[:,54])


    if snsr.any():
        axs[3,2].plot(t, snsr[:,49:52], lw=my_lw, color='blue', alpha=my_alpha)
        axs[3,2].plot(t, snsr[:,55:58], lw=my_lw, color='red', alpha=my_alpha)
    plt.gca().set_color_cycle(None)
    axs[3,2].plot(t, est_snsr[:,49:52], ls=my_ls, alpha=1.0)
    axs[3,2].plot(t, est_snsr[:,55:58], ls=my_ls, alpha=1.0)
    #axs[1,0].fill_between(t, est_snsr+std_snsr, est_snsr-std_snsr, ls=my_ls, alpha=1.0)
    axs[3,2].set_title('torque')
    for col in range(49,52):
        axs[3,2].fill_between(t, est_snsr[:,col]+std_snsr[:,col],
                est_snsr[:,col]-std_snsr[:,col], edgecolor='none', alpha=0.1)
    for col in range(55,58):
        axs[3,2].fill_between(t, est_snsr[:,col]+std_snsr[:,col],
                est_snsr[:,col]-std_snsr[:,col], edgecolor='none', alpha=0.1)
    print "R Torque Max X:", np.max(est_snsr[:,49]), np.min(est_snsr[:,49])
    print "R Torque Max Y:", np.max(est_snsr[:,50]), np.min(est_snsr[:,50])
    print "R Torque Max Z:", np.max(est_snsr[:,51]), np.min(est_snsr[:,51])
    print "L Torque Max X:", np.max(est_snsr[:,55]), np.min(est_snsr[:,55])
    print "L Torque Max Y:", np.max(est_snsr[:,56]), np.min(est_snsr[:,56])
    print "L Torque Max Z:", np.max(est_snsr[:,57]), np.min(est_snsr[:,57])



axs[0,2].plot(t, p_time, ls=my_ls, alpha=1.0, label="Prediction")
axs[0,2].set_title('timings')
axs[0,2].plot(t, c_time, ls=my_ls, alpha=1.0, label="Correction")
axs[0,2].plot(t, c_time+p_time, ls=my_ls, alpha=1.0, label="Total")
axs[0,2].legend()

print "Prediction Timing Avg:", np.mean(p_time), np.std(p_time)
print "Correction Timing Avg:", np.mean(c_time), np.std(c_time)
print "     Total Timing Avg:", np.mean(c_time+p_time), np.std(c_time+p_time)

#axs[0,2].set_title('correct time')

#axs[1,2].legend()
#for col in range(nq):
#    axs[1,2].fill_between(t, est_qpos[:,col]+std_qpos[:,col],
#            est_qpos[:,col]-std_qpos[:,col], edgecolor='none', alpha=0.5)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.show()
