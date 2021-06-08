#!/bin/bash

num_servers=$1
num_gpus=$2
base=$3
for (( i=1; i<=$1; i++ ))
do
    port=$((i*base))
    fuser $port/tcp -k
    fuser $((port + 1))/tcp -k
    fuser $((port + 2))/tcp -k
	CUDA_VISIBLE_DEVICES=$(((i-1)%num_gpus)) $CARLA11_ROOT/CarlaUE4.sh -world-port=$port -opengl -trafficmanager-port=$((port+2)) &
done
wait
