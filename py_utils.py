import numpy as np
import pandas as pd
import sys

def get_est_data(f):
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
    
    #nq = est_qpos.shape[1]
    #nv = est_qvel.shape[1]
    #nu = est_ctrl.shape[1]
    #ns = est_snsr.shape[1]
    
    std_qpos = df.filter(regex='stddev_p').values
    std_qvel = df.filter(regex='stddev_v').values
    std_ctrl = df.filter(regex='stddev_c').values
    std_snsr = df.filter(regex='stddev_s').values
    std_qpos = np.sqrt(std_qpos) 
    std_qvel = np.sqrt(std_qvel) 
    std_ctrl = np.sqrt(std_ctrl) 
    std_snsr = np.sqrt(std_snsr) 
    
    p_time = df['predict']
    c_time = df['correct']

    return {'time':t,
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
    print t
    print mrkr.shape
    if (mrkr.shape[1] > 107):
        mrkr = mrkr.reshape(t, 16,4)
        print mrkr[0,:,3]
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

def get_real_data(f, max_t):
    df = pd.read_csv(f, sep=',')
    t = df['time']
    t = t[t<max_t]
    e = len(t)
    #if limit < len(t):
    #    e = limit

    ctrl = df.filter(regex='ctrl').values
    snsr = df.filter(regex='snsr').values
    conf = df.filter(regex='conf').values
    print snsr.shape
    
    #nu = ctrl.shape[1]
    #ns = snsr.shape[1]
    
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
        mrkr = mrkr.reshape(len(t), 16,4)
        print mrkr[0,:,3]
    else:
        mrkr = mrkr.reshape(len(t), 16,3)
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
    #return t, ctrl, conf, qpos, qvel, accl, gyro, ctct, mrkr, ps



