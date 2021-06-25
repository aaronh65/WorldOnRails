import os
import yaml
from runners import ScenarioRunner

def main(args):

    towns = {i: f'Town{i+1:02d}' for i in range(7)}
    towns.update({7: 'Town10HD'})

    # scenario = 'assets/all_towns_traffic_scenarios.json'
    scenario = 'assets/no_scenarios.json'
    #route = 'assets/routes_all.xml'
    #route = 'assets/route_training.xml'
    route_dir = f'assets/routes_{args.split}'
    route_paths = [os.path.join(route_dir, route) for route in sorted(os.listdir(route_dir))]
    bounds = list(range(0, len(route_paths), len(route_paths)//args.num_runners))
    # route = 'assets/routes_training/route_10.xml'

    args.agent = 'autoagents/collector_agents/q_collector_image' # Use 'viz_collector' for collecting pretty images
    agent_config_path = 'config.yaml'
    with open(agent_config_path, 'r') as f:
        agent_config = yaml.safe_load(f)
    agent_config_path = os.path.join(agent_config['main_data_dir'], 'agent_config.yaml')
    with open(agent_config_path, 'w') as f:
        yaml.dump(agent_config, f, default_flow_style=False, sort_keys=False)
    args.agent_config = agent_config_path

    program_config = vars(args)
    program_config_path = os.path.join(agent_config['main_data_dir'], 'program_config.yaml')
    with open(program_config_path, 'w') as f:
        yaml.dump(program_config, f, default_flow_style=False, sort_keys=False)
    #config.update(vars(args))

    #args.agent_config = 'experiments/config_nocrash.yaml'

    # args.agent = 'autoagents/collector_agents/lidar_q_collector'
    # args.agent_config = 'config_lidar.yaml'
    
    record=''
    if args.record:
        record = agent_config['main_data_dir']

    jobs = []
    for i in range(args.num_runners):
        # scenario_class = 'train_scenario' # Use 'nocrash_train_scenario' to collect NoCrash training trajs
        scenario_class = args.scenario
        town = towns.get(i, 'Town03')
        port = (i+1) * args.port
        tm_port = port + 2
        #checkpoint = f'results/{i:02d}_{args.checkpoint}'
        checkpoint = agent_config['main_data_dir']

        start = bounds[i]
        end = bounds[i+1] if i != args.num_runners-1 else len(route_paths)
        routes = route_paths[start:end]
        runner = ScenarioRunner.remote(args, scenario_class, scenario, routes, checkpoint=checkpoint, town=town, port=port, tm_port=tm_port, record=record)
        jobs.append(runner.run.remote())
    
    ray.wait(jobs, num_returns=args.num_runners)


if __name__ == '__main__':

    import argparse
    import ray
    ray.init(logging_level=40, local_mode=False, log_to_driver=True)

    parser = argparse.ArgumentParser()

    parser.add_argument('--num-runners', type=int, default=8)
    parser.add_argument('--scenario', choices=['train_scenario', 'nocrash_train_scenario'], default='train_scenario')
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--timeout', default="600.0",
                        help='Set the CARLA client timeout value in seconds')
    parser.add_argument('--split', type=str, default='training', choices=['devtest','testing','training'])

    # agent-related options
    # parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True)
    # parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')
    parser.add_argument("--track", type=str, default='MAP', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument('--record', action='store_true')    
    args = parser.parse_args()
    
    main(args)
