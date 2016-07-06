import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import argparse
import os
import glob


parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Directory of MC out files")
parser.add_argument("-p", "--doPlot", action="store_true", help="Print plots")
parser.add_argument("-w", "--whole", action="store_true", help="print out entire covariance")

args = parser.parse_args()
dir = args.dir

if args.whole:
	np.set_printoptions(threshold='nan')
	print("Printing total covar")
else: 
	print("Printing truncated covar")


filelist = glob.glob(dir+"*_*.csv")
#print(filelist)

datatmp = pd.read_csv(filelist[0], sep=',')
#print(datatmp)
timelength = datatmp.shape[0]
datalength = (datatmp.shape[1] - 2) / 3 + 1

realdata = np.zeros((timelength,datalength,len(filelist)))
estdata = np.zeros((timelength,datalength,len(filelist)))

#print(datatmp)

for i in range(len(filelist)):
	df = pd.read_csv(filelist[i], sep=',')

	t = df.filter(regex='time').values
	t = np.array(t)

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

	# std_qpos = df.filter(regex='stddev_p').values
	# std_qvel = df.filter(regex='stddev_v').values
	# std_ctrl = df.filter(regex='stddev_c').values
	# std_snsr = df.filter(regex='stddev_s').values
	# std_qpos = np.sqrt(std_qpos) 
	# std_qvel = np.sqrt(std_qvel) 
	# std_ctrl = np.sqrt(std_ctrl) 
	# std_snsr = np.sqrt(std_snsr) 

	realtmp = np.hstack((t, qpos, qvel, ctrl, snsr))
	esttmp = np.hstack((t, est_qpos, est_qvel, est_ctrl, est_snsr))

	realdata[:,:, i] = realtmp
	estdata[:,:,i] = esttmp


	
avgreal = realdata.mean(axis = 2)
avgest = estdata.mean(axis = 2)

#print(avgreal)
#print(avgdata.shape)

#Calculate covariance
#3-D matrix: covariance by time
realcovars = np.zeros((datalength - 1, datalength - 1, timelength))
estcovars = np.zeros((datalength - 1, datalength - 1, timelength))

for i in range(timelength):
	realcovars[:, :, i] = np.cov(realdata[i, 1:datalength, :])
	estcovars[:, :, i] = np.cov(estdata[i, 1:datalength, :])

#Calculate standard deviation from covariance
std_qpos = np.zeros((timelength, nq))
std_qvel = np.zeros((timelength, nv))
std_ctrl = np.zeros((timelength, nu))
std_snsr = np.zeros((timelength, ns))

for i in range(timelength):
	std_qpos[i, :] = np.sqrt(estcovars[0:nq, 0:nq, i].diagonal())
	std_qvel[i, :] = np.sqrt(estcovars[nq:nq+nv, nq:nq+nv, i].diagonal())
	std_ctrl[i, :] = np.sqrt(estcovars[nq+nv:nq+nv+nu, nq+nv:nq+nv+nu, i].diagonal())
	std_snsr[i, :] = np.sqrt(estcovars[nq+nv+nu:nq+nv+nu+ns, nq+nv+nu:nq+nv+nu+ns, i].diagonal())

#Calculate RMS error
realerror = np.zeros((timelength, datalength))
esterror = np.zeros((timelength, datalength))
for i in range(len(filelist)):
	realerror = realerror + np.square(realdata[:,:,i] - avgreal)
	esterror = esterror + np.square(estdata[:, :, i] - avgest)
realerror = np.sqrt(realerror)
esterror = np.sqrt(esterror)


#print(data[3,:,:])
# print(realcovars[:,:,10])
# print(avgreal[10,:])
# print(estcovars[:,:,1])
# print("Root mean square error")
# print(realerror)
# print("dim of covar: ")
# print(realcovars[:,:,0].shape)
# print(estcovars[:,:,0].shape)


#Do plot of avg data and std dev around the mean
if args.doPlot:
	fig, axs = plt.subplots(2, 2, sharex=False)

	my_ls = '--'
	my_lw = 5
	my_alpha = 0.1
	if qpos.any():
	    axs[0,0].plot(t, qpos, lw=my_lw, alpha=my_alpha)
	plt.gca().set_color_cycle(None) # reset color cycle
	axs[0,0].plot(t, est_qpos, ls=my_ls, alpha=1.0)
	#axs[1,0].fill_between(t, est_qpos+std_qpos, est_qpos-std_qpos, ls=my_ls, alpha=1.0)
	axs[0,0].set_title('qpos')
	print est_qpos.shape
	#for col in range(nq):
	#col = 0
	#axs[0,0].fill_between(t[:,0], est_qpos[:,col]+std_qpos[:,col], 
	#    	est_qpos[:,col]-std_qpos[:,col], edgecolor='none', alpha=0.1)

	if qvel.any():
	    axs[0,1].plot(t, qvel, lw=my_lw, alpha=my_alpha)
	plt.gca().set_color_cycle(None)
	axs[0,1].plot(t, est_qvel, ls=my_ls, alpha=1.0)
	#axs[1,0].fill_between(t, est_qvel+std_qvel, est_qvel-std_qvel, ls=my_ls, alpha=1.0)
	axs[0,1].set_title('qvel')
	for col in range(nv):
	    axs[0,1].fill_between(t[:,0], est_qvel[:,col]+std_qvel[:,col],
	            est_qvel[:,col]-std_qvel[:,col], edgecolor='none', alpha=0.1)

	if ctrl.any():
	    axs[1,0].plot(t, ctrl, lw=my_lw, alpha=my_alpha)
	plt.gca().set_color_cycle(None)

	if est_ctrl.any():
	    axs[1,0].plot(t[:,0], est_ctrl, ls=my_ls, alpha=1.0)
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
	    axs[1,1].fill_between(t[:,0], est_snsr[:,col]+std_snsr[:,col],
	            est_snsr[:,col]-std_snsr[:,col], edgecolor='none', alpha=0.1)

	plt.show()




