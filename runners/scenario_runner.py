import os
import ray
from pathlib import Path
from copy import deepcopy
from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
from leaderboard.utils.statistics_manager import StatisticsManager

#@ray.remote(num_cpus=1./8, num_gpus=1./4, max_restarts=100, max_task_retries=-1)
@ray.remote(num_cpus=1., num_gpus=1., max_restarts=100, max_task_retries=-1)
class ScenarioRunner():
    def __init__(self, args, scenario_class, scenario, route, checkpoint='simulation_results.json', town=None, port=2000, tm_port=2002, debug=False, record=''):
        args = deepcopy(args)

        # Inject args
        args.scenario_class = scenario_class
        args.town = town
        args.port = port
        args.trafficManagerPort = tm_port
        args.scenarios = scenario
        args.debug = debug
        args.checkpoint = checkpoint
        args.record = record 
        args.routes = route

        self.args = args

    def run(self):
        for r in self.args.routes:
            print(r)
            args = deepcopy(self.args)
            args.routes = r
            rname = r.split('/')[-1].split('.')[0]
            ckpt = Path(os.path.join(args.checkpoint, 'logs'))
            ckpt.mkdir(parents=True,exist_ok=True)
            args.checkpoint = str(ckpt / f'{rname}.json')
            runner = LeaderboardEvaluator(args, StatisticsManager())
            ret = runner.run(args)

        return ret
