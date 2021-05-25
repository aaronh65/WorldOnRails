#!/bin/bash

for (( i=1; i<=$1; i++ ))
do
    port=$((i*$2))
	echo $port
    fuser $port/tcp -k
    fuser $((port + 1))/tcp -k
    fuser $((port + 2))/tcp -k
	$CARLA_ROOT/CarlaUE4.sh -world-port=$port -opengl -trafficmanager-port=$((port+2)) &
done
wait
