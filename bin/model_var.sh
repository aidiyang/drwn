#!/bin/bash

d_noise="1e-4"
#data_f="straight_walk.csv"
data_f="walk3.csv"


./viewer -m ../models/darwin.xml    -r1 -f $data_f -R0 -d $d_noise -i -1 -t 1 -o none.csv >/dev/null &
./viewer -m ../models/darwin_t.xml  -r1 -f $data_f -R0 -d $d_noise -i -1 -t 1 -o t_noise.csv >/dev/null &
./viewer -m ../models/darwin_m.xml  -r1 -f $data_f -R0 -d $d_noise -i -1 -t 1 -o m_noise.csv >/dev/null &
./viewer -m ../models/darwin_tm.xml -r1 -f $data_f -R0 -d $d_noise -i -1 -t 1 -o tm_noise.csv >/dev/null &

wait
ipython ../rms_compare.py $data_f none.csv t_noise.csv m_noise.csv tm_noise.csv
