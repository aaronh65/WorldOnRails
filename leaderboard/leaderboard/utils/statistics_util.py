import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from collections import deque
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from srunner.scenariomanager.traffic_events import TrafficEventType

colors = sns.color_palette("Paired")

PENALTY_COLLISION_PEDESTRIAN = 0.50
PENALTY_COLLISION_VEHICLE = 0.60
PENALTY_COLLISION_STATIC = 0.65
PENALTY_TRAFFIC_LIGHT = 0.70
PENALTY_STOP = 0.80

collision_types = [
        TrafficEventType.COLLISION_PEDESTRIAN, 
        TrafficEventType.COLLISION_VEHICLE,
        TrafficEventType.COLLISION_STATIC]

penalty_dict = {
        TrafficEventType.COLLISION_PEDESTRIAN  : PENALTY_COLLISION_PEDESTRIAN,
        TrafficEventType.COLLISION_VEHICLE  : PENALTY_COLLISION_VEHICLE,
        TrafficEventType.COLLISION_STATIC  : PENALTY_COLLISION_STATIC,
        TrafficEventType.TRAFFIC_LIGHT_INFRACTION  : PENALTY_TRAFFIC_LIGHT,
        TrafficEventType.STOP_INFRACTION  : PENALTY_STOP
        }

string_dict = {
        TrafficEventType.COLLISION_PEDESTRIAN  : f'hit ped ({PENALTY_COLLISION_PEDESTRIAN}x)',
        TrafficEventType.COLLISION_VEHICLE  : f'hit vehicle ({PENALTY_COLLISION_VEHICLE}x)',
        TrafficEventType.COLLISION_STATIC  : f'hit static ({PENALTY_COLLISION_STATIC}x)',
        TrafficEventType.TRAFFIC_LIGHT_INFRACTION  : f'ran light ({PENALTY_TRAFFIC_LIGHT}x)',
        TrafficEventType.STOP_INFRACTION  : f'ran stop ({PENALTY_STOP}x)',
        }

def plot_performance(score_route_list, infraction_list, checkpoint, scenario_triggerer, route_var_name_class_lookup, tol=1e-4):
        
    fig = plt.gcf()
    fig.set_size_inches(12,8)
    ax = plt.gca()
    
    # compute penalties
    infraction_list = sorted(infraction_list, key=lambda x: x[0])
    inf_time_mult = deque([(time, penalty_dict[itype]) for time, itype in infraction_list])
    score_penalty = [1.0] * len(score_route_list)
    for i in range(1, len(score_penalty)):

        score_penalty[i] = score_penalty[i-1]
        if len(inf_time_mult) == 0:
            continue

        # check for active infraction and apply penalty if so
        inf_time, penalty = inf_time_mult[0]
        if abs(i*0.05 - inf_time) < tol or i*0.05 - inf_time >= 0.05:
            score_penalty[i] = score_penalty[i-1]*penalty
            inf_time_mult.popleft()

    # compute driving scores and reduce to 2 Hz
    score_composed_list = np.multiply(score_penalty, score_route_list) # 20 Hz
    score_composed_plot = score_composed_list[::10] # 2 Hz
    score_route_plot = score_route_list[::10]

    # plot scores
    x_plot = np.arange(len(score_composed_plot)) * 0.5 # 2 Hz
    plt.plot(x_plot, score_route_plot, label='route completion', color=colors[0])
    plt.plot(x_plot, score_composed_plot, label='driving score', color=colors[2])

    # useful for plotting
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    increment = ymax/30
    repeat = 8

    # plot infraction times
    for i, (time, itype) in enumerate(infraction_list):
        offset = increment*(i%repeat + 1) # extra 1 so text shows up below the top of the plot
        plt.vlines(time, ymin, ymax, linestyles='dashed', alpha=0.5, color='red')
        plt.text(time+0.2, ymax-offset, string_dict[itype])

    # plot scenario trigger times
    scenarios = scenario_triggerer._triggered_scenarios
    times = scenario_triggerer._triggered_scenarios_times
    lookup = route_var_name_class_lookup
    for i, (time, route_var_name) in enumerate(zip(times, scenarios)):
        offset = increment*(i%repeat)
        plt.vlines(time, ymin, ymax, linestyles='dashed', alpha=0.5, color='purple')
        plt.text(time+0.2, ymin+offset+ymax/60, lookup[route_var_name])
    
    # label axes and format ticks
    plt.xlabel('Game time')
    plt.ylabel('Score (%)')
    def format_ticks(value, tick_number):
        minute = int(value/60)
        return f'{minute:02d}:00'
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_ticks))
    ax.xaxis.set_minor_locator(MultipleLocator(15))
    ax.tick_params(which='both', direction='in')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    
    # finish up and save
    SAVE_ROOT = os.environ.get('SAVE_ROOT', 0)
    ROUTE_NAME = os.environ.get('ROUTE_NAME', 0)
    REPETITION = os.environ.get('REPETITION', 0)
    title = f'{ROUTE_NAME}/{REPETITION}'
    title = title.replace('_', ' ')
    plt.title(title)
    plt.legend(frameon=False, loc='lower right')
    save_path = Path(f'{SAVE_ROOT}/plots/{ROUTE_NAME}')
    save_path.mkdir(parents=True,exist_ok=True)
    plt.savefig(str(save_path / f'{REPETITION}.png'))
    plt.clf()


