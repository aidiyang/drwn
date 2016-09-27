
import sys, os

from pyqtgraph.Qt import QtGui
from pyqtgraph.Qt import QtGui, QtCore
import pandas as pd
import numpy as np
import pyqtgraph as pg

app = QtGui.QApplication([])

#pg.setConfigOption('background', 'w')
#pg.setConfigOption('foreground', 'k')
# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)
win = pg.GraphicsWindow(title="Lets look at the datas")
win.resize(1800,1000)
win.setWindowTitle('pyqtgraph example: Plotting')

###############################################
if len(sys.argv) > 1:
    f = sys.argv[1]
else:
    f = "out.csv"

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

colors = [
   (  0,   114,   189, 255),
   (217,    83,    25, 255),
   (237,   177,    32, 255),
   (126,    47,   142, 255),
   (119,   172,    48, 255),
   ( 77,   190,   238, 255),
   (162,    20,    47, 255)]


brushes = [
   (  0,   114,   189, 155),
   (217,    83,    25, 155),
   (237,   177,    32, 155),
   (126,    47,   142, 155),
   (119,   172,    48, 155),
   ( 77,   190,   238, 155),
   (162,    20,    47, 155)]


pens = [
   pg.mkPen((  0,   114,   189, 155), width=5), 
   pg.mkPen((217,    83,    25, 155), width=5),
   pg.mkPen((237,   177,    32, 155), width=5),
   pg.mkPen((126,    47,   142, 155), width=5),
   pg.mkPen((119,   172,    48, 155), width=5),
   pg.mkPen(( 77,   190,   238, 155), width=5),
   pg.mkPen((162,    20,    47, 155), width=5)]


def fill_up(plot_num, t, mean, stddev, n):
    curves_plus = [plot_num.plot(x=np.array(t), y=(mean[:,i]+stddev[:,i]),
        pen=brushes[col%7]) for i in range(n)]
    curves_minus= [plot_num.plot(x=np.array(t), y=(mean[:,i]-stddev[:,i]),
        pen=brushes[col%7]) for i in range(n)]
    fills = [pg.FillBetweenItem(curves_plus[i], curves_minus[i], brushes[i%7])
            for i in range(n)]
    for f in fills:
        plot_num.addItem(f)



nq = qpos.shape[1]
p1 = win.addPlot(title="Qpos")
for col in range(nq):
    p1.plot(t, qpos[:,col], pen=colors[col%7])
    p1.plot(t, est_qpos[:,col], pen=colors[col%7], width=3)

fill_up(p1, t, est_qpos, std_qpos, nq)

win.nextRow()
nv = qvel.shape[1]
p2 = win.addPlot(title="Qvel")
for col in range(nv):
    p2.plot(t, qvel[:,col], pen=colors[col%7])
    p2.plot(t, est_qvel[:,col], pen=colors[col%7], width=3)

fill_up(p2, t, est_qvel, std_qvel, nv)

win.nextRow()

nu = ctrl.shape[1]
p3 = win.addPlot(title="Ctrl")
for col in range(nu):
    p3.plot(t, ctrl[:,col], pen=colors[col%7])
    p3.plot(t, est_ctrl[:,col], pen=colors[col%7], width=3)

#fill_up(p3, t, est_ctrl, std_ctrl, nu)


win.nextRow()

my_pen = pg.mkPen('y', width=3);
ns = snsr.shape[1]
p4 = win.addPlot(title="Sensors")
for col in range(ns):
    p4.plot(t, snsr[:,col], pen=colors[col%7])
    p4.plot(t, est_snsr[:,col], pen=pens[col%7], width=30)

fill_up(p4, t, est_snsr, std_snsr, ns)


win.nextRow()

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
