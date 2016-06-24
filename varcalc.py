import numpy as np
import pandas as pd
import sys
import os
import glob


assert len(sys.argv) > 1

#np.set_printoptions(threshold='nan')	enables printing of entire matrix

dir = sys.argv[1]
if (len(sys.argv) > 2):
	if (sys.argv[2] == "true"):
		np.set_printoptions(threshold='nan')
		print("Printing total covar")
	else: 
		print("Printing truncated covar")
else:
	print("Printing truncated covar")


	

filelist = glob.glob(dir+"*_*.csv")
#print(filelist)

datatmp = pd.read_csv(filelist[0], sep=',')
timelength = datatmp.shape[0]
datalength = datatmp.shape[1] - 1

data = np.zeros((timelength,datalength,len(filelist)))

for i in range(len(filelist)):
	df = pd.read_csv(filelist[i], sep=',')
	time = df.filter(regex='time').values
	qpos = df.filter(regex='qpos').values
	qvel = df.filter(regex='qvel').values
	ctrl = df.filter(regex='ctrl').values
	snsr = df.filter(regex='snsr').values

	tmp = np.hstack((time, qpos, qvel, ctrl, snsr))
	data[:,:, i] = tmp

	
avgdata = data.mean(axis = 2)

#print(avgdata)
#print(avgdata.shape)

#4-D matrix: covariance by time by simulation
covars = np.zeros((datalength, datalength, timelength))

#print(data[:,:,0])

for i in range(timelength):
	covars[:,:, i] = np.cov(data[i, :, :])

x = np.array([[1, 1], [1, 1]])
#print(np.cov(x))

	
error = np.zeros((timelength, datalength))
for i in range(len(filelist)):
	error = error + np.square(data[:,:,i] - avgdata)
error = np.sqrt(error)


#print(data[3,:,:])
#print(covars[:,:,3])
print("Root mean square error")
print(error)
print("dim of covar: ")
print(covars[:,:,0].shape)



