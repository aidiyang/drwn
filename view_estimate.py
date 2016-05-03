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

p_time = df['predict']
c_time = df['correct']


fig, axs = plt.subplots(2, 3, sharex=False)

my_ls = '--'
my_lw = 10
my_alpha = 0.1
axs[0,0].plot(t, qpos, lw=my_lw, alpha=my_alpha)
plt.gca().set_color_cycle(None) # reset color cycle
axs[0,0].plot(t, est_qpos, ls=my_ls, alpha=1.0)
axs[0,0].set_title('qpos')

axs[0,1].plot(t, qvel, lw=my_lw, alpha=my_alpha)
plt.gca().set_color_cycle(None)
axs[0,1].plot(t, est_qvel, ls=my_ls, alpha=1.0)
axs[0,1].set_title('qvel')

axs[1,0].plot(t, ctrl, lw=my_lw, alpha=my_alpha)
plt.gca().set_color_cycle(None)
axs[1,0].plot(t, est_ctrl, ls=my_ls, alpha=1.0)
axs[1,0].set_title('ctrl')

axs[1,1].plot(t, snsr, lw=my_lw, alpha=my_alpha)
plt.gca().set_color_cycle(None)
axs[1,1].plot(t, est_snsr, ls=my_ls, alpha=1.0)
axs[1,1].set_title('sensors')

axs[0,2].plot(t, p_time, ls=my_ls, alpha=1.0)
axs[0,2].set_title('predict time')
axs[1,2].plot(t, c_time, ls=my_ls, alpha=1.0)
axs[1,2].set_title('correct time')

plt.show()
