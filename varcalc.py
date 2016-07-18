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
parser.add_argument("-s", "--plotSensor", action="store_true", help="Plot sensor data")

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
datalength = (datatmp.shape[1] - 1) / 5 + 1

realdata = np.zeros((timelength, datalength, len(filelist)))
estdata = np.zeros((timelength, datalength, len(filelist)))
ekfdata = np.zeros((timelength, datalength, len(filelist)))
#4-D matrix: Covariance by time by simulation (file)
covardata = np.zeros((datalength - 1, datalength - 1, timelength, len(filelist)))
ekf_covardata = np.zeros((datalength - 1, datalength - 1, timelength, len(filelist)))

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

	ekf_qpos = df.filter(regex='ekf_p').values
	ekf_qvel = df.filter(regex='ekf_v').values
	ekf_ctrl = df.filter(regex='ekf_c').values
	ekf_snsr = df.filter(regex='ekf_s').values

	nq = est_qpos.shape[1]
	nv = est_qvel.shape[1]
	nu = est_ctrl.shape[1]
	ns = est_snsr.shape[1]

	covar_qpos = df.filter(regex='stddev_p').values
	covar_qvel = df.filter(regex='stddev_v').values
	covar_ctrl = df.filter(regex='stddev_c').values
	covar_snsr = df.filter(regex='stddev_s').values

	ekf_covar_qpos = df.filter(regex='ekfdev_p').values
	ekf_covar_qvel = df.filter(regex='ekfdev_v').values
	ekf_covar_ctrl = df.filter(regex='ekfdev_c').values
	ekf_covar_snsr = df.filter(regex='ekfdev_s').values

	realtmp = np.hstack((t, qpos, qvel, ctrl, snsr))
	esttmp = np.hstack((t, est_qpos, est_qvel, est_ctrl, est_snsr))
	ekftmp = np.hstack((t, ekf_qpos, ekf_qvel, ekf_ctrl, ekf_snsr))
	covartmp = np.hstack((covar_qpos, covar_qvel, covar_ctrl, covar_snsr))
	ekf_covartmp = np.hstack((ekf_covar_qpos, ekf_covar_qvel, ekf_covar_ctrl, ekf_covar_snsr))

	realdata[:, :, i] = realtmp
	estdata[:, :, i] = esttmp
	ekfdata[:, :, i] = ekftmp
	for j in range(timelength):
		covardata[:, :, j, i] = np.diag(covartmp[j, :])
		ekf_covardata[:, :, j, i] = np.diag(ekf_covartmp[j, :])

avgreal = np.mean(realdata, axis = 2)
avgest = np.mean(estdata, axis = 2)
avgekf = np.mean(ekfdata, axis = 2)
avgcovar = np.mean(covardata, axis = 3) #UKF covariance: Covariance by time (3-D)
avg_ekfcovar = np.mean(ekf_covardata, axis = 3)\

#Calculate covariance
#3-D matrix: covariance by time
realcovars = np.zeros((datalength - 1, datalength - 1, timelength))
estcovars = np.zeros((datalength - 1, datalength - 1, timelength))
ekfcovars = np.zeros((datalength - 1, datalength - 1, timelength))

for i in range(timelength):
	realcovars[:, :, i] = np.cov(realdata[i, 1:datalength, :])
	estcovars[:, :, i] = np.cov(estdata[i, 1:datalength, :])
	ekfcovars[:, :, i] = np.cov(ekfdata[i, 1:datalength, :])

#Calculate standard deviation from covariance
std_qpos = np.zeros((timelength, nq))
std_qvel = np.zeros((timelength, nv))
std_ctrl = np.zeros((timelength, nu))
std_snsr = np.zeros((timelength, ns))
ekf_std_qpos = np.zeros((timelength, nq))
ekf_std_qvel = np.zeros((timelength, nv))
ekf_std_ctrl = np.zeros((timelength, nu))
ekf_std_snsr = np.zeros((timelength, ns))

for i in range(timelength):
	std_qpos[i, :] = np.sqrt(estcovars[0:nq, 0:nq, i].diagonal())
	std_qvel[i, :] = np.sqrt(estcovars[nq:nq+nv, nq:nq+nv, i].diagonal())
	std_ctrl[i, :] = np.sqrt(estcovars[nq+nv:nq+nv+nu, nq+nv:nq+nv+nu, i].diagonal())
	std_snsr[i, :] = np.sqrt(estcovars[nq+nv+nu:nq+nv+nu+ns, nq+nv+nu:nq+nv+nu+ns, i].diagonal())
	ekf_std_qpos[i, :] = np.sqrt(ekfcovars[0:nq, 0:nq, i].diagonal())
	ekf_std_qvel[i, :] = np.sqrt(ekfcovars[nq:nq+nv, nq:nq+nv, i].diagonal())
	ekf_std_ctrl[i, :] = np.sqrt(ekfcovars[nq+nv:nq+nv+nu, nq+nv:nq+nv+nu, i].diagonal())
	ekf_std_snsr[i, :] = np.sqrt(ekfcovars[nq+nv+nu:nq+nv+nu+ns, nq+nv+nu:nq+nv+nu+ns, i].diagonal())

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

	ekf_qpos = avgekf[:,1:nq+1]
	ekf_qvel = avgekf[:,1+nq:nq+nv+1]
	ekf_ctrl = avgekf[:,1+nq+nv:nq+nv+nu+1]
	ekf_snsr = avgekf[:,1+nq+nv+nu:nq+nv+nu+ns+1]

	fig, axs = plt.subplots(2, 2, sharex=False)

	#qpos/qvel 2 is z 
	my_ls = '--'
	my_lw = 4
	my_alpha = 0.5

	#Plot qpos[0] (x position)
	if qpos.any():
	    axs[0,0].plot(t, qpos[:,0], lw=my_lw, alpha=my_alpha, label = 'Real qpos')
	#plt.gca().set_color_cycle(None) # reset color cycle
	axs[0,0].plot(t, est_qpos[:,0], ls=my_ls, c = 'red', alpha=1.0, label = 'UKF qpos')
	axs[0,0].plot(t, ekf_qpos[:,0], ls='-', c = 'black', alpha=1.0, label = 'EKF qpos')
	axs[0,0].set_title('x qpos')
	handles, labels = axs[0,0].get_legend_handles_labels()
	axs[0,0].legend(handles, labels)
	#Plot UKF covar lines	
	axs[0,0].fill_between(t[:,0], est_qpos[:,0]+std_qpos[:,0], 
	   	est_qpos[:,0]-std_qpos[:,0], edgecolor='black', alpha=0.1)

	#Plot qpos[1] (z position)
	if qpos.any():
	    axs[1,0].plot(t, qpos[:,1], lw=my_lw, alpha=my_alpha, label = 'Real qpos')
	#plt.gca().set_color_cycle(None) # reset color cycle
	axs[1,0].plot(t, est_qpos[:,1], ls=my_ls, c = 'red', alpha=1.0, label = 'UKF qpos')
	axs[1,0].plot(t, ekf_qpos[:,1], ls='-', c = 'black', alpha=1.0, label = 'EKF qpos')
	axs[1,0].set_title('z qpos')
	handles, labels = axs[1,0].get_legend_handles_labels()
	axs[1,0].legend(handles, labels)
	#Plot UKF covar lines	
	axs[1,0].fill_between(t[:,0], est_qpos[:,1]+std_qpos[:,1], 
	   	est_qpos[:,1]-std_qpos[:,1], edgecolor='black', alpha=0.1)

	#Plot qvel[0] (x velocity)
	if qvel.any():
	    axs[0,1].plot(t, qvel[:, 0], lw=my_lw, alpha=my_alpha, label = 'Real qvel')
	#plt.gca().set_color_cycle(None)
	axs[0,1].plot(t, est_qvel[:, 0], ls=my_ls, c = 'red', alpha=1.0, label = 'UKF qvel')
	axs[0,1].plot(t, ekf_qpos[:, 0], ls='-', c = 'black', alpha=1.0, label = 'EKF qvel')
	axs[0,1].set_title('x qvel')
	handles, labels = axs[0,1].get_legend_handles_labels()
	axs[0,1].legend(handles, labels)
	for col in range(nv):
		axs[0,1].fill_between(t[:,0], est_qvel[:,0]+std_qvel[:,0],
	            est_qvel[:,0]-std_qvel[:,0], edgecolor='none', alpha=0.1)

	#Plot qvel[1] (z velocity)
	if qvel.any():
	    axs[1,1].plot(t, qvel[:, 1], lw=my_lw, alpha=my_alpha, label = 'Real qvel')
	#plt.gca().set_color_cycle(None)
	axs[1,1].plot(t, est_qvel[:, 1], ls=my_ls, c = 'red', alpha=1.0, label = 'UKF qvel')
	axs[1,1].plot(t, ekf_qpos[:, 1], ls='-', c = 'black', alpha=1.0, label = 'EKF qvel')
	axs[1,1].set_title('x qvel')
	handles, labels = axs[1,1].get_legend_handles_labels()
	axs[1,1].legend(handles, labels)
	for col in range(nv):
		axs[0,1].fill_between(t[:,0], est_qvel[:,1]+std_qvel[:,1],
	            est_qvel[:,1]-std_qvel[:,1], edgecolor='none', alpha=0.1)

	plt.show()

#TODO: Figure out color cycle
if args.plotSensor:

	snsr = avgreal[:,1+nq+nv+nu:nq+nv+nu+ns+1]
	est_snsr = avgest[:,1+nq+nv+nu:nq+nv+nu+ns+1]
	ekf_snsr = avgekf[:,1+nq+nv+nu:nq+nv+nu+ns+1]

	if snsr.any():
	    plt.plot(t, snsr, lw=4, alpha=0.5, label='Real sensor')
	plt.gca().set_color_cycle(None)
	plt.plot(t, est_snsr, ls='--', alpha=1.0, label='UKF sensor')
	plt.plot(t, ekf_snsr, ls='-', c = 'black', alpha=1.0, label = 'EKF sensor')
	#axs[1,0].fill_between(t, est_snsr+std_snsr, est_snsr-std_snsr, ls=my_ls, alpha=1.0)
	plt.title('sensors')
	#handles, labels = plt.get_legend_handles_labels()
	plt.legend()
	for col in range(ns):
	    plt.fill_between(t[:,0], est_snsr[:,col]+std_snsr[:,col],
	            est_snsr[:,col]-std_snsr[:,col], edgecolor='none', alpha=0.1)

	plt.show()

if args.doQpos:

	for i in range(len(filelist)):
		qpos = realdata[:,1:nq+1, i]
		est_qpos = estdata[:, 1:nq+1, i]
		ekf_qpos = ekfdata[:, 1:nq+1, i]
		plt.plot(t, qpos, 'b--')
		plt.plot(t, ekf_qpos, 'r--')
		plt.title("Simulation qpos")
		#plt.gca().set_color_cycle(None) # reset color cycle
		#plt.plot(t, est_qpos, ls='--', alpha=1.0)
		plt.show()

#TODO: Add ekf error as well. Need to calc error above
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
