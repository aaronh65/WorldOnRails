#####################
# this script is usually used on a remote cluster

import os, sys, time, shutil
import yaml, argparse
import subprocess, psutil, traceback
from datetime import datetime
from collections import deque
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('-D', '--debug', action='store_true')
parser.add_argument('-G', '--gpus', type=int, default=1)
parser.add_argument('--agent', type=str, default='lbc/src/auto_pilot.py')
parser.add_argument('--split', type=str, default='devtest', 
        choices=['devtest','testing','training','debug'])
parser.add_argument('--repetitions', type=int, default=1)

parser.add_argument('--data_root', type=str, default='/data/aaronhua')
parser.add_argument('--id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
parser.add_argument('--port', type=int, default=2000)
parser.add_argument('--save_data', action='store_true')
parser.add_argument('--save_debug', action='store_true')
parser.add_argument('--config_path', type=str)
args = parser.parse_args()

assert args.data_root != '/data/aaronhua', 'should not do heavy I/O to /data'

# specific for multiprocessing and running multiple servers
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

project_root = os.environ['PROJECT_ROOT']

# save root
tokens = args.agent.split('.')[0].split('/')
appr, algo = tokens[0], tokens[-1]
prefix = '/data/aaronhua/wor/data' if args.save_data else '/data/aaronhua/leaderboard/benchmark'
suffix = f'debug/{args.id}' if args.debug else args.id
save_root = Path(f'{prefix}/wor/{algo}/{suffix}')
save_root.mkdir(parents=True,exist_ok=True)
(save_root / 'plots').mkdir(exist_ok=True)
(save_root / 'logs').mkdir(exist_ok=True)
(save_root / 'data').mkdir(exist_ok=True)

# agent-specific config
try:
        
    # launch CARLA servers
    carla_procs = list()
    worker_procs = list()
    gpus = list(range(args.gpus))
    port_map = {gpu: (args.port*(gpu+1), args.port*(gpu+1)+2) for i, gpu in enumerate(gpus)}

    # agent-specific configurations
    config_path = f'{project_root}/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config['project_root'] = project_root
    config['save_root'] = str(save_root)
    config['save_data'] = args.save_data
    config['save_debug'] = args.save_debug
    config['split'] = args.split
    config['repetitions'] = args.repetitions

    track = 'SENSORS'

    config_path = f'{save_root}/config.yml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # route paths
    route_dir = f'{project_root}/assets/routes_{args.split}'
    routes = [route.split('.')[0] for route in sorted(os.listdir(route_dir)) if route.endswith('.xml')]

    split_len = {'devtest':4,'testing':26,'training':50,}
    routes = deque(list(range(split_len[args.split])))
    gpu_free = [True] * len(gpus) # True if gpu can be used for next leaderboard process
    gpu_proc = [None] * len(gpus) # tracks which gpu has which process

    # main testing loop
    while len(routes) > 0 or not all(gpu_free):

        # check for finished leaderboard runs
        for i, (free, proc) in enumerate(zip(gpu_free, gpu_proc)):
            # check if gpus has a valid process and if it's done
            if proc and proc.poll() is not None: 
                gpu_free[i] = True
                gpu_proc[i] = None

        # sleep and goto next iter if busy
        if True not in gpu_free or len(routes) == 0:
            time.sleep(5)
            continue
        
        # make image + performance plot dirs
        gpu = gpu_free.index(True)
        wp, tp = port_map[gpu]
        routenum = routes.popleft()
        route_name = f'route_{routenum:02d}'

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["SAVE_ROOT"] = str(save_root)
        env["TRACK"] = track
        env["WORLD_PORT"] = str(wp)
        env["TM_PORT"] = str(tp)
        env["AGENT"] = args.agent
        env["SPLIT"] = args.split
        env["ROUTE_NAME"] = route_name
        env["REPETITIONS"] = str(args.repetitions)

        # run command
        cmd = f'bash {project_root}/scripts/run_leaderboard.sh &> {save_root}/logs/AGENT_{route_name}.txt'
        print(f'running {cmd} on {args.split}/{route_name} for {args.repetitions} repetitions')
        worker_procs.append(subprocess.Popen(cmd, env=env, shell=True))

        gpu_free[gpu] = False
        gpu_proc[gpu] = worker_procs[-1]

except KeyboardInterrupt:
    print('detected keyboard interrupt')

except Exception as e:
    traceback.print_exc()

print('shutting down processes...')
for proc in carla_procs + worker_procs:
    try:
        kill(proc.pid)
    except:
        continue
print('done')
