#!/bin/bash

nonoise="1e-12"
d_noise="1e-4"
ee="0.4"
data_f="clean_straight_walk.csv"
#data_f="clean_fallen_3_short.csv"
#data_f="walk3.csv"

./viewer -m ../models/darwin.xml    -e $ee -r1 -f $data_f -R0 -d $nonoise -i -1 -t 1 -o nonoise.csv >/dev/null &

./viewer -m ../models/darwin.xml    -e $ee -r1 -f $data_f -R0 -d $d_noise -i -1 -t 1 -o c_noise.csv >/dev/null &
./viewer -m ../models/darwin_t.xml  -e $ee -r1 -f $data_f -R0 -d $nonoise -i -1 -t 1 -o t_noise.csv >/dev/null &
./viewer -m ../models/darwin_m.xml  -e $ee -r1 -f $data_f -R0 -d $nonoise -i -1 -t 1 -o m_noise.csv >/dev/null &

#wait
#echo "4 done"

./viewer -m ../models/darwin_t.xml  -e $ee -r1 -f $data_f -R0 -d $d_noise -i -1 -t 1 -o tc_noise.csv >/dev/null &
./viewer -m ../models/darwin_m.xml  -e $ee -r1 -f $data_f -R0 -d $d_noise -i -1 -t 1 -o mc_noise.csv >/dev/null &

./viewer -m ../models/darwin_tm.xml -e $ee -r1 -f $data_f -R0 -d $nonoise -i -1 -t 1 -o tm_noise.csv >/dev/null &
./viewer -m ../models/darwin_tm.xml -e $ee -r1 -f $data_f -R0 -d $d_noise -i -1 -t 1 -o tmc_noise.csv >/dev/null &

wait
echo "8 done"

ipython ../rms_compare.py $data_f nonoise.csv c_noise.csv t_noise.csv m_noise.csv tc_noise.csv mc_noise.csv tm_noise.csv tmc_noise.csv
