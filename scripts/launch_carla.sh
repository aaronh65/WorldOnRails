#!/bin/bash

for (( i=1; i<=$1; i++ ))
do
    port=$((i*$2))
    fuser $port/tcp -k
    fuser $((port + 1))/tcp -k
    fuser $((port + 2))/tcp -k
    CUDA_VISIBLE_DEVICES=$((i%$3)) $CARLA_ROOT/CarlaUE4.sh -world-port=$port -opengl -trafficmanager-port=$((port+2)) &
done
wait
