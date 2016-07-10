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
parser.add_argument("-w", "--whole", action="store_true", help="Print out entire covariance")
parser.add_argument("-q", "--doQpos", action="store_true", help="Plot qpos for each simulation")
parser.add_argument("-e", "--plotError", action="store_true", help="Plot RMS error")

args = parser.parse_args()
dir = args.dir

if args.whole:
	np.set_printoptions(threshold='nan')
	print("Printing total covar")
else: 
	print("Printing truncated covar")


filelist = glob.glob(dir+"*_*.csv")

datatmp = pd.read_csv(filelist[0], sep=',')
timelength = datatmp.shape[0]
datalength = (datatmp.shape[1] - 2) / 3 + 1

realdata = np.zeros((timelength, datalength, len(filelist)))
estdata = np.zeros((timelength, datalength, len(filelist)))
#4-D matrix: Covariance by time by simulation (file)
covardata = np.zeros((datalength - 1, datalength - 1, timelength, len(filelist)))

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

	covar_qpos = df.filter(regex='stddev_p').values
	covar_qvel = df.filter(regex='stddev_v').values
	covar_ctrl = df.filter(regex='stddev_c').values
	covar_snsr = df.filter(regex='stddev_s').values

	realtmp = np.hstack((t, qpos, qvel, ctrl, snsr))
	esttmp = np.hstack((t, est_qpos, est_qvel, est_ctrl, est_snsr))
	covartmp = np.hstack((covar_qpos, covar_qvel, covar_ctrl, covar_snsr))

	realdata[:, :, i] = realtmp
	estdata[:, :, i] = esttmp
	for j in range(timelength):
		covardata[:, :, j, i] = np.diag(covartmp[j, :])

avgreal = np.mean(realdata, axis = 2)
avgest = np.mean(estdata, axis = 2)
avgcovar = np.mean(covardata, axis = 3) #UKF covariance: Covariance by time (3-D)

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
#NOTE: RMS error has dimensions timelength by datalength (RMS error includes time unlike NEES error)
#Can change/fix later. Probably make RMS not include time, since no point/sense adding time into covar
realerror = np.zeros((timelength))
esterror = np.zeros((timelength))
for j in range(timelength):
	for i in range(len(filelist)):
		realerror[j] = realerror[j] + np.dot(realdata[j,:,i] - avgreal[j, :], realdata[j,:,i] - avgreal[j, :])
		esterror[j] = esterror[j] + np.dot(estdata[j, :, i] - avgreal[j, :], estdata[j, :, i] - avgreal[j, :])
realerror = np.sqrt(realerror)
esterror = np.sqrt(esterror)

#Calculate NEES error
#NOTE: NEES error uses covariance which does not include time, so NEES also doesn't include time 
#(dimensions are timelength by datalength - 1)
neeserr = np.zeros((timelength))
for j in range(timelength):
	for i in range(len(filelist)):
		tmp = realdata[j, 1:datalength, i] - avgreal[j, 1:datalength]
		#FIX: Covariance is singular, adding identity
		#print(tmp)
		neeserr[j] = neeserr[j] + np.dot(tmp, np.dot(np.linalg.inv(avgcovar[:, :, i]+np.identity(datalength-1)), np.transpose(tmp)))
	neeserr[j] = neeserr[j] / len(filelist)

#TODO: Figure out way to print covariance

#print(data[3,:,:])
#print(realcovars[:,:,:])
# print(avgreal[10,:])
# print(estcovars[:,:,1])
#print("Root mean square error")
#print(esterror)
# print("dim of covar: ")
# print(realcovars[:,:,0].shape)
# print(estcovars[:,:,0].shape)

#Do plot of avg data and std dev around the mean
if args.doPlot:
	
	qpos = avgreal[:,1:nq+1]
	qvel = avgreal[:,1+nq:nq+nv+1]
	ctrl = avgreal[:,1+nq+nv:nq+nv+nu+1]
	snsr = avgreal[:,1+nq+nv+nu:nq+nv+nu+ns+1]

	est_qpos = avgest[:,1:nq+1]
	est_qvel = avgest[:,1+nq:nq+nv+1]
	est_ctrl = avgest[:,1+nq+nv:nq+nv+nu+1]
	est_snsr = avgest[:,1+nq+nv+nu:nq+nv+nu+ns+1]

	fig, axs = plt.subplots(2, 2, sharex=False)

	#qpos/qvel 2 is z 
	my_ls = '--'
	my_lw = 4
	my_alpha = 0.5

	if qpos.any():
	    axs[0,0].plot(t, qpos, lw=my_lw, alpha=my_alpha)
	plt.gca().set_color_cycle(None) # reset color cycle
	axs[0,0].plot(t, est_qpos, ls=my_ls, alpha=1.0)
	axs[0,0].set_title('qpos')
	for col in range(nq):	
		axs[0,0].fill_between(t[:,0], est_qpos[:,col]+std_qpos[:,col], 
		   	est_qpos[:,col]-std_qpos[:,col], edgecolor='none', alpha=0.1)

	if qvel.any():
	    axs[0,1].plot(t, qvel, lw=my_lw, alpha=my_alpha)
	plt.gca().set_color_cycle(None)
	axs[0,1].plot(t, est_qvel, ls=my_ls, alpha=1.0)
	axs[0,1].set_title('qvel')
	for col in range(nv):
	#col = j;
		axs[0,1].fill_between(t[:,0], est_qvel[:,col]+std_qvel[:,col],
	            est_qvel[:,col]-std_qvel[:,col], edgecolor='none', alpha=0.1)

	if ctrl.any():
	    axs[1,0].plot(t, ctrl, lw=my_lw, alpha=my_alpha)
	plt.gca().set_color_cycle(None)

	if est_ctrl.any():
	    axs[1,0].plot(t, est_ctrl, ls=my_ls, alpha=1.0)
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

if args.doQpos:

	for i in range(len(filelist)):
		qpos = realdata[:,1:nq+1, i]
		est_qpos = estdata[:, 1:nq+1, i]
		plt.plot(t, qpos, 'b--')
		plt.title("Simulation qpos")
		#plt.gca().set_color_cycle(None) # reset color cycle
		#plt.plot(t, est_qpos, ls='--', alpha=1.0)
		plt.show()

if args.plotError:

	# RMSqpos = esterror[:,1:nq+1]
	# RMSqvel = esterror[:,1+nq:nq+nv+1]
	# RMSctrl = esterror[:,1+nq+nv:nq+nv+nu+1]
	# RMSsnsr = esterror[:,1+nq+nv+nu:nq+nv+nu+ns+1]

	fig, axs = plt.subplots(2, sharex=False)

	axs[0].plot(t, esterror)
	axs[0].set_title('RMS error')
	axs[1].plot(t, neeserr)
	axs[1].set_title('NEES error')
	plt.show()

	#qpos/qvel 2 is z 
	# my_ls = '--'
	# my_lw = 2
	# my_alpha = 0.5

	# if qpos.any():
	#     axs[0,0].plot(t, RMSqpos, lw=my_lw, alpha=my_alpha)
	# plt.gca().set_color_cycle(None) # reset color cycle
	# axs[0,0].set_title('RMS error qpos')

	# if qvel.any():
	#     axs[0,1].plot(t, RMSqvel, lw=my_lw, alpha=my_alpha)
	# plt.gca().set_color_cycle(None)
	# axs[0,1].set_title('RMS error qvel')
	
	# if ctrl.any():
	#     axs[1,0].plot(t, RMSctrl, lw=my_lw, alpha=my_alpha)
	# plt.gca().set_color_cycle(None)

	# axs[1,0].set_title('RMS error ctrl')

	# if snsr.any():
	#     axs[1,1].plot(t, RMSsnsr, lw=my_lw, alpha=my_alpha)
	# plt.gca().set_color_cycle(None)
	# axs[1,1].set_title('RMS error sensors')

	# plt.figure()
	# plt.plot(t, neeserr)
	# plt.title("NEES error")
	# plt.show()
