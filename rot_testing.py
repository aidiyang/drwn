import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt

from numpy.linalg import norm

f = 'raw.csv'
r = 'rot.csv'
if len(sys.argv) < 2:
    print "python [data.csv] [rot.csv]"
    sys.exit

f = sys.argv[1]
#r = sys.argv[2]

df = pd.read_csv(f, sep=',')

t = df['time']
ctrl = df.filter(regex='ctrl').values
snsr = df.filter(regex='snsr').values
print snsr.shape

nu = ctrl.shape[1]
ns = snsr.shape[1]

qpos = snsr[:,0:20]
qvel = snsr[:,20:40]
accl = snsr[:,40:43]
gyro = snsr[:,43:46]
ctct = snsr[:,46:58]
mrkr = snsr[:,58:]
if (snsr.shape[1] > 107):
    mrkr = mrkr.reshape(len(t), 16,4)
    print mrkr[0,:,3]
else:
    mrkr = mrkr.reshape(len(t), 16,3)

ps = np.copy(mrkr[:,:,0:3])
print mrkr[0,:,3]

print "Qpos size:", qpos.shape
print "Qvel size:", qvel.shape
print "Accl size:", accl.shape
print "Gyro size:", gyro.shape
print "Ctct size:", ctct.shape
print "Mrkr size:", mrkr.shape
print "Ps   size:", ps.shape

rot=np.array([[ 0.00694805,    0.997699,  -0.0674413],
    [   0.999955, -0.00649409,  0.00694805],
    [-0.00649409,   0.0674865,    0.997699]])

rot=np.array([[ 1,    0,  0],
    [   0, 1,  0],
    [0, 0, 1]])
print "\tTo XY plan rotation Matrix from File:\n",rot
l1=np.array([
    -0.220229,
    0.501774,
    -0.00216621])
r1=np.array([
    -0.182336,
    0.365584,
    0.00134366])
l2=np.array([
    0.032274,
    0.561796,
    0.0144842])

def angle(v1, v2):
    return np.arccos(np.clip(np.dot(v1, v2)/norm(v1)/norm(v2), -1, 1))
#return np.arccos(np.dot(v1, v2)/norm(v1)/norm(v2))

def normalize(v):
    return v/norm(v)

def rot_matrix(v1, v2):
    dot = np.dot(v1, v2)
    cross = norm(np.cross(v1, v2))
    return np.array([
        [dot, -1*cross, 0],
        [cross, dot, 0],
        [0, 0, 1]]);

m_normal = np.cross(r1-l1, l2-l1) # normal vector of marker space's plane
m_normal = normalize(m_normal);
s_normal = np.array([0,0,1]) # normal of simulator's plane
s_normal = normalize(s_normal);
normal_theta = angle(m_normal, s_normal)
print "Angle between marker's plane's normal and sim's normal:", np.rad2deg(normal_theta)

new_rot = rot_matrix(m_normal, s_normal)
rot = new_rot

print "Recalculated Rotation to Plane Matrix:\n", new_rot


#TODO calculate variance from no-walk data

s_ps_zero = np.array(
  [[-0.0440,  0.0560, 0.3918], 
   [-0.0750, -0.0540, 0.3228],
   [ 0.0040,  0.0450, 0.3873],
   [ 0.0350,  0.0180, 0.3478],
   [-0.0440, -0.0560, 0.3918],
   [-0.0750,  0.0540, 0.3228],
   [ 0.0040, -0.0450, 0.3873],
   [ 0.0350, -0.0180, 0.3478],
   [-0.0490, -0.0723, 0.0162],
   [-0.0490,  0.0723, 0.0162],
   [ 0.0390, -0.0723, 0.0117],
   [ 0.0390,  0.0723, 0.0117],
   [ 0.0710, -0.1324, 0.3141],
   [ 0.0960, -0.1517, 0.3063],
   [ 0.0710,  0.1326, 0.3151],
   [ 0.0960,  0.1519, 0.3073]])

s_ps_walk = np.array([
   [-0.0246,  0.0560, 0.3758],
   [-0.0711, -0.0540, 0.3161],
   [ 0.0210,  0.0450, 0.3601],
   [ 0.0417,  0.0180, 0.3144],
   [-0.0246, -0.0560, 0.3758],
   [-0.0711,  0.0540, 0.3161],
   [ 0.0210, -0.0450, 0.3601],
   [ 0.0417, -0.0180, 0.3144],
   [-0.0795, -0.0743, 0.0169],
   [-0.0795,  0.0743, 0.0169],
   [ 0.0084, -0.0742, 0.0117],
   [ 0.0084,  0.0742, 0.0117],
   [ 0.0061, -0.1081, 0.2651],
   [ 0.0205, -0.1174, 0.2375],
   [ 0.0161,  0.1082, 0.2655],
   [ 0.0337,  0.1176, 0.2399]])

def r2s_from_chest_markers(vec_r, vec_s):
    c1 = 7; # that wacky sensor ordering...
    c2 = 3;
    #vec_r[2] = 0
    #vec_s[2] = 0
    
    vec_z = np.array([1,0,0]) #normalize(np.array([1,1,0]))
    vec_z = normalize(vec_z)
    vec_r = normalize(vec_r)
    vec_s = normalize(vec_s)
    
    # TODO deal with 4 mm difference in sensor norms...
    print "Chest Vector from markers, norm:\n", vec_r, norm(vec_r)
    print "Chest Vector from Simultr, norm:\n", vec_s, norm(vec_s), s_ps_walk[c1,:], s_ps_walk[c2,:]
    
    angle_diff = angle(vec_r, vec_s)
    
    print "Angle difference: ", np.rad2deg(angle_diff), ", radian:", angle_diff
    
    a1 = angle(vec_r, vec_z)
    a2 = angle(vec_s, vec_z)
    
    print "Angles from independent point:", np.rad2deg(a1), np.rad2deg(a2)
    
    if (a2 > a1):
        r2s = np.array([
            [np.cos(angle_diff),  np.sin(angle_diff), 0],
            [-1*np.sin(angle_diff), np.cos(angle_diff), 0],
            [0, 0, 1]]);
    else:
        r2s = np.array([
            [np.cos(angle_diff), -1*np.sin(angle_diff), 0],
            [np.sin(angle_diff),  np.cos(angle_diff), 0],
            [0, 0, 1]]);
    
    return r2s

def r2s_from_yaw(vec_r, vec_s):
    # TODO keep track of the directionality
    #print "yaw vectors:\n", vec_r, "\n", vec_s
    vec_z = normalize(np.array([1,0,0]))
    vec_r[2] = 0
    vec_s[2] = 0
    vec_r = normalize(vec_r)
    vec_s = normalize(vec_s)
    a1 = angle(vec_r, vec_z)
    a2 = angle(vec_s, vec_z)

    angle_diff = angle(vec_r, vec_s)
    if (np.isfinite(angle_diff)):
        if (a2 > a1):
            r2s = np.array([
                [np.cos(angle_diff),  np.sin(angle_diff), 0],
                [-1*np.sin(angle_diff), np.cos(angle_diff), 0],
                [0, 0, 1]]);
        else:
            r2s = np.array([
                [np.cos(angle_diff), -1*np.sin(angle_diff), 0],
                [np.sin(angle_diff),  np.cos(angle_diff), 0],
                [0, 0, 1]]);
        

        #r2s = rot_matrix(vec_r, vec_s)
        print "angle: ", np.rad2deg(angle(vec_r, vec_s))
        return r2s, 1 
    else:
        return 0, 0


s = np.zeros((16,3))
count = 100
c = np.zeros((16,1))
for i in range(0,count):
    for j in range(0,16):
        if (mrkr[i,j,3] > 3.99):
            s[j,:] += rot.dot(mrkr[i,j,0:3])
            c[j] = c[j] + 1 

print "Sum:"
s = s / c
for i in range(0,16):
    print i, s[i,:], c[i]
print "\n"
cc = c/count;

print "Averaged Initial marker position:\n", s
#c1 = 9; # that wacky sensor ordering...
#vec_r = rot.dot(s[c1,:])
#vec_s = s_ps_walk[c1,:]
#r2s = r2s_from_yaw(vec_r, vec_s)

c1 = 7; # that wacky sensor ordering...
c2 = 3;
#vec_r = rot.dot(mrkr[t1,c1,0:3]) - rot.dot(mrkr[t1,c2,0:3])
v1=np.copy(s[c1,:])
v2=np.copy(s[c2,:])
vec_s = s_ps_walk[c1,:] - s_ps_walk[c2,:] # TODO set z value to be 0 for these two?
v1[2]=0
v2[2]=0
vec_s[2]=0
#vec_r = rot.dot(s[c1,:]) - rot.dot(s[c2,:])
vec_r = rot.dot(v1) - rot.dot(v2)
r2s = r2s_from_chest_markers(vec_r, vec_s)

#r2s = np.zeros((3,3))
#count = 0;
#for c1 in range(0, 16):
#    vec_r = rot.dot(s[c1,:])
#    vec_s = s_ps_walk[c1,:]
#    
#    r2, c = r2s_from_yaw(vec_r, vec_s)
#    if (c):
#        r2s += r2
#        count += c
#    #print r2s
#
#r2s = r2s / count

#r2s=np.array([[ 1,    0,  0],
#    [   0, 1,  0],
#    [0, 0, 1]])

print "Final Rotation Matrix:\n", r2s

v1 = np.array([1,1,0])
v2 = np.array([0,1,0])

#vr = r2s_from_yaw(normalize(v1), normalize(v2))
vr, c = r2s_from_yaw(v1, v2)

print "norms v1:", norm(v1), " v2:", norm(v2) 
print "test angle:", np.rad2deg(angle(v1,v2))
print "test rot:\n", vr 
print " mult v1:", vr.dot(v1)
print " mult v2:", vr.dot(v2)
print "norms rotated v1:", norm(vr.dot(v1)), " v2:", norm(vr.dot(v2))


###### lets use our rotations
print "initial markers v real:\n",
t2 = len(t) - 500
offset = np.zeros((16,3))
print "# distance marker\n",

t1=0
print "offset:"
for i in range(0,16):
    #vec=r2s.dot(rot.dot(mrkr[t1,i,0:3]))
    vec=r2s.dot(rot.dot(s[i,0:3]))
    offset[i,:] = vec - s_ps_walk[i,:]
    #vec2=r2s.dot(rot.dot(mrkr[t2,i,0:3])) - offset[i,:]
    vec2=r2s.dot(rot.dot(s[i,0:3])) - offset[i,:]
    print i, vec2, offset[i,:], mrkr[t1,i,3]
    #print i, norm(mrkr[t1,i,0:3]-mrkr[t2,i,0:3]), mrkr[t1, i, 0:3], mrkr[t2, i, 0:3]

print "offset var 1", np.var(offset, axis=0)
print "\n"
cutoff = 0.99
print cc[:,0]>cutoff
print offset[cc[:,0]>cutoff]
print "offset var 2", np.var(offset[cc[:,0]>cutoff], axis=0)
print "offsetstddev", np.std(offset[cc[:,0]>cutoff], axis=0)
print "offset  mean",np.mean(offset[cc[:,0]>cutoff], axis=0)

rot_traj = np.copy(ps)
for time in range(0, t.shape[0]):
    for i in range(0,16):
        vec2=r2s.dot(rot.dot(mrkr[time,i,0:3])) - offset[i,:]
        #print i, vec2, s_ps_walk[i,:]
        rot_traj[time, i, :] = vec2;


new_mrkr = np.copy(mrkr)

for time in range(0, t.shape[0]):
    for i in range(0,16):
        new_mrkr[time, i, 0:3] = rot_traj[time, i, :]
new_mrkr = new_mrkr.reshape(len(t), 16*4)
snsr[:,58:] = new_mrkr

#t = df['time']
#ctrl = df.filter(regex='ctrl').values
#snsr = df.filter(regex='snsr').values

#new_data = np.hstack((t[:,None], ctrl, snsr))
#head = 'time,'+'ctrl,'*nu+'snsr,'*ns
#np.savetxt('rotated_'+f, new_data, delimiter=',', header=head, fmt='%.8f',
#        comments='')
# 
#df = pd.read_csv('rotated_'+f, sep=',')
#t = df['time']
#print snsr.shape
#
#n_mrkr = snsr[:,58:]
#n_mrkr = n_mrkr.reshape(len(t), 16,4)
#new_ps = n_mrkr[:,:,0:3]
new_ps = mrkr[:,:,0:3]


############### plot

fig, axs = plt.subplots(3, 3, sharex=False)

my_ls = '--'
my_lw = 2
my_alpha = 0.9
axs[0,0].plot(t, new_ps[:,:,0], lw=my_lw, alpha=my_alpha)
axs[0,0].set_title('x-axis')
axs[0,0].set_ylim([-1, 1])

axs[0,1].plot(t, new_ps[:,:,1], lw=my_lw, alpha=my_alpha)
axs[0,1].set_title('y-axis')
axs[0,1].set_ylim([-1, 1])

axs[0,2].plot(t, new_ps[:,:,2], lw=my_lw, alpha=my_alpha)
axs[0,2].set_title('z-axis')
axs[0,2].set_ylim([-1, 1])

axs[1,0].plot(t, ps[:,:,0], lw=my_lw, alpha=my_alpha)
axs[1,0].set_title('raw x-axis')
axs[1,0].set_ylim([-1, 1])

axs[1,1].plot(t, ps[:,:,1], lw=my_lw, alpha=my_alpha)
axs[1,1].set_title('raw y-axis')
axs[1,1].set_ylim([-1, 1])

axs[1,2].plot(t, ps[:,:,2], lw=my_lw, alpha=my_alpha)
axs[1,2].set_title('raw z-axis')
axs[1,2].set_ylim([-1, 1])

axs[2,0].plot(t, mrkr[:,:,3], lw=my_lw, alpha=my_alpha)
axs[2,1].plot(t, mrkr[:,:,3], lw=my_lw, alpha=my_alpha)
axs[2,2].plot(t, mrkr[:,:,3], lw=my_lw, alpha=my_alpha)
axs[2,0].set_ylim([-2, 15])
axs[2,1].set_ylim([-2, 15])
axs[2,2].set_ylim([-2, 15])

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.show()

