#!/bin/bash


dd="1e-4"
f="clean_straight_walk.csv"
#f="clean_fallen_3.csv"

./viewer -m ../models/darwin_tm.xml -r1 -f $f -R0 -d $dd -i -1 -t 1 -o all_snsr.csv >/dev/null &
./viewer -m ../models/d_no_accel.xml -r1 -f $f -R0 -d $dd -i -1 -t 1 -o no_accel.csv >/dev/null &
./viewer -m ../models/d_no_gyro.xml -r1 -f $f -R0 -d $dd -i -1 -t 1 -o no_gyro.csv >/dev/null &
./viewer -m ../models/d_no_force.xml -r1 -f $f -R0 -d $dd -i -1 -t 1 -o no_force.csv >/dev/null &
./viewer -m ../models/d_no_torque.xml -r1 -f $f -R0 -d $dd -i -1 -t 1 -o no_torque.csv >/dev/null &
./viewer -m ../models/d_no_jpos.xml -r1 -f $f -R0 -d $dd -i -1 -t 1 -o no_jpos.csv >/dev/null &
./viewer -m ../models/d_no_jvel.xml -r1 -f $f -R0 -d $dd -i -1 -t 1 -o no_jvel.csv >/dev/null &

wait
ipython ../loo_compare.py $f
