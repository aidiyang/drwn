import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import argparse
import os
import glob

if len(sys.argv) < 2:
    print("python [data1.csv] [data2.csv]")
    sys.exit

input_file = sys.argv[1]
print("Input file:", input_file)

df = pd.read_csv(input_file, sep=',')
timelength = df.shape[0]
print("timelength", timelength)

t = df.filter(regex='time').values
print(type(t))

print("t shape", t.shape)
print("t[1]", t[1])

qpos = df.filter(regex='qpos').values
qvel = df.filter(regex='qvel').values
ctrl = df.filter(regex='ctrl').values
snsr = df.filter(regex='snsr').values

ekf_qpos = df.filter(regex='EKF_p').values
ekf_qvel = df.filter(regex='EKF_v').values
ekf_ctrl = df.filter(regex='EKF_c').values
ekf_snsr = df.filter(regex='EKF_s').values

PF_qpos = df.filter(regex='PF_p').values
PF_qvel = df.filter(regex='PF_v').values
PF_ctrl = df.filter(regex='PF_c').values
PF_snsr = df.filter(regex='PF_s').values

kNN_qpos = df.filter(regex='kNN_p').values
kNN_qvel = df.filter(regex='kNN_v').values
kNN_ctrl = df.filter(regex='kNN_c').values
kNN_snsr = df.filter(regex='kNN_s').values

nq = qpos.shape[1]
nv = qvel.shape[1]
nu = ctrl.shape[1]
ns = snsr.shape[1]

realstate = np.zeros((timelength, nq+nv))
EKFstate = np.zeros((timelength, nq+nv))
PFstate = np.zeros((timelength, nq+nv))
kNNstate = np.zeros((timelength, nq+nv))

realstate = np.hstack((qpos, qvel))
EKFstate = np.hstack((ekf_qpos, ekf_qvel))
PFstate = np.hstack((PF_qpos, PF_qvel))
kNNstate = np.hstack((kNN_qpos, kNN_qvel))

EKF_err = realstate - EKFstate
PF_err = realstate - PFstate
kNN_err = realstate - kNNstate

EKF_err = np.power(EKF_err, 2)
PF_err = np.power(PF_err, 2)
kNN_err = np.power(kNN_err, 2)

EKF_err = np.sum(EKF_err, 1)
PF_err = np.sum(PF_err, 1)
kNN_err = np.sum(kNN_err, 1)

EKF_err = np.power(EKF_err, .5)
PF_err = np.power(PF_err, .5)
kNN_err = np.power(kNN_err, .5)




plt.plot(t, EKF_err, 'r', t, PF_err, 'b', t, kNN_err, 'g')
plt.ylim([0, .6])
plt.show()
#Plot stuff