import numpy as np
import pandas as pd
import sys

def max_min(snsr):
    return np.max(snsr), np.min(snsr)

def get_est_data(f):
    df = pd.read_csv(f, sep=',')
    
    #s = 0
    time = np.copy(df['time'].values)
    
    qpos = np.copy(df.filter(regex='qpos').values)
    qvel = np.copy(df.filter(regex='qvel').values)
    ctrl = np.copy(df.filter(regex='ctrl').values)
    snsr = np.copy(df.filter(regex='snsr').values)
    
    est_qpos = np.copy(df.filter(regex='est_p').values)
    est_qvel = np.copy(df.filter(regex='est_v').values)
    est_ctrl = np.copy(df.filter(regex='est_c').values)
    est_snsr = np.copy(df.filter(regex='est_s').values)
    
    std_qpos = np.copy(df.filter(regex='stddev_p').values)
    std_qvel = np.copy(df.filter(regex='stddev_v').values)
    std_ctrl = np.copy(df.filter(regex='stddev_c').values)
    std_snsr = np.copy(df.filter(regex='stddev_s').values)
    std_qpos = np.sqrt(std_qpos) 
    std_qvel = np.sqrt(std_qvel) 
    std_ctrl = np.sqrt(std_ctrl) 
    std_snsr = np.sqrt(std_snsr) 
    
    p_time = np.copy(df['predict'])
    c_time = np.copy(df['correct'])

    return {'time':time,
            'qpos':qpos, 
            'qvel':qvel,
            'ctrl':ctrl,
            'snsr':snsr,
            'est_qpos':est_qpos,
            'est_qvel':est_qvel,
            'est_ctrl':est_ctrl,
            'est_snsr':est_snsr,
            'std_qpos':std_qpos,
            'std_qvel':std_qvel,
            'std_ctrl':std_ctrl,
            'std_snsr':std_snsr,
            'p_time':p_time,
            'c_time':c_time}


def snsr_breakout(snsr):
    qpos = snsr[:,0:20]
    qvel = snsr[:,20:40]
    accl = snsr[:,40:43]
    gyro = snsr[:,43:46]
    ctct = snsr[:,46:58]
    mrkr = snsr[:,58:]
    t = qpos.shape[0]
    #print t
    #print mrkr.shape
    if (mrkr.shape[1] > 107):
        mrkr = mrkr.reshape(t, 16,4)
        #print mrkr[0,:,3]
    else:
        mrkr = mrkr.reshape(t, 16,3)
    ps = np.copy(mrkr[:,:,0:3])

    return {'qpos':qpos, 
            'qvel':qvel,
            'accl':accl, 
            'gyro':gyro, 
            'ctct':ctct,
            'mrkr':mrkr,
            'ps':ps}

def get_pure_mrkr(f): # from outputs that are not mine
    df = pd.read_csv(f, sep=',')
    mrkr = df.filter(regex='mrkr').values
    # all x all y all z
    conf = df.filter(regex='conf').values
    print "Mrkr shape", mrkr.shape
    print "Conf shape", conf.shape

    return {'mrkr': mrkr, 'conf': conf}

def get_real_data(f, max_t):
    df = pd.read_csv(f, sep=',')
    t = df['time']
    #t = t[t<=max_t]
    if max_t < 0:
        max_t = len(t)

    t = t[0:max_t]
    e = len(t)
    #if limit < len(t):
    #    e = limit

    ctrl = df.filter(regex='ctrl').values
    snsr = df.filter(regex='snsr').values
    conf = df.filter(regex='conf').values
    #print snsr.shape
    
    t = t[0:e]
    ctrl = ctrl[0:e,:]
    conf = conf[0:e,:]
    qpos = snsr[0:e,0:20]
    qvel = snsr[0:e,20:40]
    accl = snsr[0:e,40:43]
    gyro = snsr[0:e,43:46]
    ctct = snsr[0:e,46:58]
    mrkr = snsr[0:e,58:]
    if (mrkr.shape[1] > 107):
        mrkr = mrkr.reshape(e, 16,4)
        #print mrkr[0,:,3]
    else:
        mrkr = mrkr.reshape(e, 16,3)
    ps = np.copy(mrkr[:,:,0:3])

    return {'time':t,
            'ctrl':ctrl, 
            'conf':conf, 
            'qpos':qpos, 
            'qvel':qvel,
            'accl':accl, 
            'gyro':gyro, 
            'ctct':ctct,
            'mrkr':mrkr,
            'ps':ps,
            'conf':conf}

def dist_diff_whole_run(x1, x2):
    # one rms value for the entire trajectory
    y = np.sum(np.square(x1-x2), axis=0)
    return y


def dist_diff_v_time(x1, x2, c, conf):
    # one rms value for the entire trajectory
    #y = np.sqrt(np.sum(np.sum(np.square(x1-x2), axis=2), axis=1))
    y = np.zeros((x1.shape[0], 1))
    for t in range(x1.shape[0]):
        count = 0
        for i in range(16):
            if c[t,i] > conf and c[t,i] < 10:
                dist = np.sum(np.square(x1[t,i,:]-x2[t,i,:]))
                y[t, 0] += dist
                count += 1
        y[t, 0] = np.sqrt(y[t, 0]/count)

    return y

def dist_diff_v_time_limited(x1, x2, c, conf, cols):
    # one rms value for the entire trajectory
    #y = np.sqrt(np.sum(np.sum(np.square(x1-x2), axis=2), axis=1))
    y = np.zeros((x1.shape[0], 1))
    for t in range(x1.shape[0]):
        count = 0
        for i in cols:
            if c[t,i] > conf and c[t,i] < 10:
                dist = np.sum(np.square(x1[t,i,:]-x2[t,i,:]))
                #print x1[t,i,:], "::", x2[t,i,:], "==", dist
                y[t, 0] += dist
                count += 1
        y[t, 0] = np.sqrt(y[t, 0]/count)

    return y

def clean_mrkr_data(time, mrkr, c, conf, vel_limit, min_t):
    T = mrkr.shape[0]
    N = 16 # num markers
    new_c = np.copy(c)
    new_c[0,:] = -1
    last = np.zeros((16)) # remember the last good values
    for t in range(1, T):
        if time[t] > min_t:
            for n in range(N):
                if c[t,n] > conf and c[t,n] < 10:
                    l = last[n]
                    d = mrkr[t,n,:]-mrkr[l,n,:]
                    vel = d.dot(d) / (time[t] - time[l])
                    if vel > vel_limit:
                        new_c[t,n] = -1 # don't trust things that move too quick
                    else:
                        last[n] = t
                else:
                    new_c[t,n] = c[t,n]
        else:
            last[:] = t
            new_c[t,:] = -1

    return new_c
        
def clean_ctct_data(ctct, first):
    i = np.mean(ctct[0:first,:], axis=0)
    # z-axis values don't offset
    i[2] = 0
    i[8] = 0
    new_c = np.copy(ctct) - i #apply offset

    return new_c


