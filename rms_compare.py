import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt

import py_utils as util

if len(sys.argv) > 2:
    hw_f = sys.argv[1]

print "Hardware File:", hw_f
hw = util.get_real_data(hw_f, -1)

fig, axs = plt.subplots(1, 1, sharex=False, figsize=(15,10))
my_lw = 2
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

#cols = [1, 2, 6, 7, 11, 12] # walk 3
#cols = [1, 2] # walk3 full
print hw['ps'][0,:,:]
init_delta = hw['ps'][0,:,:] - init_spos
hw['ps'] = hw['ps'][:,:,:] - init_delta
print hw['ps'][0,:,:]



names = ['No Noise', 'Ctrl Noise', 'Time Noise', 'Mass Noise', 'Time/Control',
        'Mass/Control', 'Time/Mass', 'Time/Mass/Control']

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
        lw=my_lw, alpha=my_alpha, label=names[f-2])


ymin, ymax = plt.ylim()
if ymax > 1:
    ymax = 0.5
    plt.ylim(ymin, ymax)

axs.set_title('RMSE for Noise Sources')
axs.set_xlabel('Time')
axs.set_ylabel('RMSE (Meters)')
axs.grid(True)
axs.legend()

plt.legend(bbox_to_anchor=(0.4, 0.9), bbox_transform=plt.gcf().transFigure)

plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

plt.show()

