import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt

import py_utils as util

if len(sys.argv) > 2:
    hw_f = sys.argv[1]

print "Hardware File:", hw_f
hw = util.get_real_data(hw_f, -1)

fig, axs = plt.subplots(1, 1, sharex=False)
my_lw = 2
my_alpha = 0.9
conf_lvl = 0
conf = hw['conf']
my_ls = ['-','--','-.',':']

#conf[:,0] = -1
#conf[:,1] = -1

new_c = util.clean_mrkr_data(hw['time'], hw['ps'], conf, 3, 0.01)

#cols = [2,4,6,7,8,11,12] # straight walk
cols = [2] # straight walk full
#cols = [1, 2, 6, 7, 11, 12] # walk 3
#cols = [1, 2] # walk3 full

for f in range(2, len(sys.argv)):
    r = sys.argv[f]
    es = util.get_est_data(r)
    t = es['time']
    ss = util.snsr_breakout(es['est_snsr'])

    
    #rms = util.dist_diff_v_time(ss['ps'], hw['ps'], conf, conf_lvl)
    #rms = util.dist_diff_v_time(ss['ps'], hw['ps'], new_c, conf_lvl)
    rms = util.dist_diff_v_time_limited(ss['ps'], hw['ps'], new_c, conf_lvl,
            cols)
    
    print "rms shape:", rms.shape
    
    idx = (f-2)%len(my_ls)
    print "ls:", idx
    axs.plot(t, rms, ls=my_ls[idx],
        lw=my_lw, alpha=my_alpha, label=r)


ymin, ymax = plt.ylim()
if ymax > 1:
    ymax = 0.75
    plt.ylim(ymin, ymax)

axs.set_title('Time vs RMS of estimator v phasespace')
axs.legend()
plt.legend(bbox_to_anchor=(0.5, 0.8), bbox_transform=plt.gcf().transFigure)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

plt.show()

