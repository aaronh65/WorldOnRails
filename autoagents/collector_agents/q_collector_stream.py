import os
import math
import yaml
import lmdb
import numpy as np
import torch
import wandb
import carla
import random
import string

from torch.distributions.categorical import Categorical
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from utils import visualize_obs, _numpy
from pathlib import Path

from rails.bellman import BellmanUpdater
from rails.models import EgoModel, CameraModel
from autoagents.waypointer import Waypointer

def get_entry_point():
    return 'QCollectorImage'

FPS = 20.
STOP_THRESH = 0.1
MAX_STOP = 500

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu=0, sigma=0.1, theta=0.1, dt=0.1, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal()
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class QCollectorImage(AutonomousAgent):

    """
    action value agent but assumes a static world
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """

        self.track = Track.MAP
        self.num_frames = 0
        
        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)
        
        for key, value in config.items():
            setattr(self, key, value)
        
        device = torch.device('cuda')
        ego_model = EgoModel(1./FPS*(self.num_repeat+1)).to(device)
        ego_model.load_state_dict(torch.load(self.ego_model_dir))
        ego_model.eval()
        BellmanUpdater.setup(config, ego_model, device=device)
        
        self.vizs      = []
        self.wide_rgbs = []
        self.narr_rgbs = []
        self.wide_sems = []
        self.narr_sems = []

        self.lbls = []
        self.locs = []
        self.rots = []
        self.spds = []
        self.cmds = []
        self.infs = []

        self.waypointer = None

        #if self.log_wandb:
        #    wandb.init(project='carla_data_phase1')

        self.noiser = OrnsteinUhlenbeckActionNoise(dt=1/FPS)
        self.prev_steer = 0
        
        self.stop_count = 0

        # for hybrid mode
        self.device = device
        self.image_model = CameraModel(config).to(self.device)
        self.image_model.load_state_dict(torch.load(self.main_model_dir))
        self.image_model.eval()

        self.steers = torch.tensor(np.linspace(-self.max_steers,self.max_steers,self.num_steers)).float().to(self.device)
        self.throts = torch.tensor(np.linspace(0,self.max_throts,self.num_throts)).float().to(self.device)

        self.lane_change_counter = 0
        self.lane_changed=None

        self.rstring = _random_string()
        self.num_infractions = 0
        #os.environ['SAVE_ROOT'] = os.path.join(self.main_data_dir, self.rstring)
        os.environ['SAVE_ROOT'] = self.main_data_dir

        self.num_samples = 0
        self.initialized = False


    def destroy(self):
        if len(self.lbls) == 0:
            return

        #self.flush_data()

    def cleanup(self):
        inf_obj = CarlaDataProvider.get_infraction_list()[-1]
        if len(self.infs) == 0:
            self.infs.append(inf_obj.get_type().value)
        else:
            self.infs[-1] = inf_obj.get_type().value
        self.flush_data(self.num_samples-1)
        with self.lmdb_env.begin(write=True) as txn:
            print(f'NUM SAMPLES {self.num_samples}')
            print(txn.put('len'.encode(), str(self.num_samples).encode()))

        self.vizs.clear()
        self.wide_rgbs.clear()
        self.narr_rgbs.clear()
        self.wide_sems.clear()
        self.narr_sems.clear()
        self.lbls.clear()
        self.locs.clear()
        self.rots.clear()
        self.spds.clear()
        self.cmds.clear()
        self.infs.clear()

        self.lmdb_env.close()

    def flush_data(self, i):
        print(f'putting data at timestep {i}')
        with self.lmdb_env.begin(write=True) as txn:
            for idx in range(len(self.camera_yaws)):
                txn.put(
                    f'wide_rgb_{idx}_{i:05d}'.encode(),
                    np.ascontiguousarray(self.wide_rgbs[i][idx]).astype(np.uint8),
                )

                txn.put(
                    f'narr_rgb_{idx}_{i:05d}'.encode(),
                    np.ascontiguousarray(self.narr_rgbs[i][idx]).astype(np.uint8),
                )

                txn.put(
                    f'wide_sem_{idx}_{i:05d}'.encode(),
                    np.ascontiguousarray(self.wide_sems[i][idx]).astype(np.uint8),
                )
                
                txn.put(
                    f'narr_sem_{idx}_{i:05d}'.encode(),
                    np.ascontiguousarray(self.narr_sems[i][idx]).astype(np.uint8),
                )

            txn.put(
                f'lbl_{i:05d}'.encode(),
                np.ascontiguousarray(self.lbls[i]).astype(np.uint8),
            )

            txn.put(
                f'loc_{i:05d}'.encode(),
                np.ascontiguousarray(self.locs[i]).astype(np.float32)
            )

            txn.put(
                f'rot_{i:05d}'.encode(),
                np.ascontiguousarray(self.rots[i]).astype(np.float32)
            )

            txn.put(
                f'spd_{i:05d}'.encode(),
                np.ascontiguousarray(self.spds[i]).astype(np.float32)
            )

            
            txn.put(
                f'cmd_{i:05d}'.encode(),
                np.ascontiguousarray(self.cmds[i]).astype(np.float32)
            )

            txn.put(
                f'inf_{i:05d}'.encode(),
                np.ascontiguousarray(self.infs[i]).astype(np.float32)
            )

    def sensors(self):
        sensors = [
            {'type': 'sensor.collision', 'id': 'COLLISION'},
            {'type': 'sensor.map', 'id': 'MAP'},
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0.0, 'z': self.camera_z, 'id': 'GPS'},
        ]
        
        # Add sensors
        for i, yaw in enumerate(self.camera_yaws):
            x = self.camera_x*math.cos(yaw*math.pi/180)
            y = self.camera_x*math.sin(yaw*math.pi/180)
            sensors.append({'type': 'sensor.stitch_camera.rgb', 'x': x, 'y': y, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'Wide_RGB_{i}'})
            sensors.append({'type': 'sensor.stitch_camera.semantic_segmentation', 'x': x, 'y': y, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'Wide_SEG_{i}'})
            sensors.append({'type': 'sensor.camera.rgb', 'x': x, 'y': y, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw,
            'width': 384, 'height': 240, 'fov': 50, 'id': f'Narrow_RGB_{i}'})
            sensors.append({'type': 'sensor.camera.semantic_segmentation', 'x': x, 'y': y, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw,
            'width': 384, 'height': 240, 'fov': 50, 'id': f'Narrow_SEG_{i}'})
            
        return sensors

    def _init(self):
        data_path = Path(self.main_data_dir) / 'data' / os.environ['ROUTE_NAME'] / os.environ['REPETITION']
        data_path.mkdir(parents=True,exist_ok=True)
        data_path = str(data_path)
        self.lmdb_env = lmdb.open(data_path, map_size=int(1e10))
        self.initialized = True

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        
        wide_rgbs = []
        narr_rgbs = []
        wide_sems = []
        narr_sems = []

        for i in range(len(self.camera_yaws)):
            
            _, wide_rgb = input_data.get(f'Wide_RGB_{i}')
            _, narr_rgb = input_data.get(f'Narrow_RGB_{i}')
            _, wide_sem = input_data.get(f'Wide_SEG_{i}')
            _, narr_sem = input_data.get(f'Narrow_SEG_{i}')
            
            wide_rgbs.append(wide_rgb[...,:3])
            narr_rgbs.append(narr_rgb[...,:3])
            wide_sems.append(wide_sem)
            narr_sems.append(narr_sem)

        _, lbl = input_data.get('MAP')
        _, col = input_data.get('COLLISION')
        _, ego = input_data.get('EGO')
        _, gps = input_data.get('GPS')

        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)
            _, _, cmd = self.waypointer.tick(gps)
        else:
            _, _, cmd = self.waypointer.tick(gps)

        yaw = ego.get('rot')[-1]
        spd = ego.get('spd')
        loc = ego.get('loc')

        
        #delta_locs, delta_yaws, next_spds = BellmanUpdater.compute_table(yaw/180*math.pi)

        # Convert lbl to rew maps
        lbl_copy = lbl.copy()
        #waypoint_rews, stop_rews, brak_rews, free = BellmanUpdater.get_reward(lbl_copy, [0,0], ref_yaw=yaw/180*math.pi)

        #waypoint_rews = waypoint_rews[None].expand(self.num_plan, *waypoint_rews.shape)
        #brak_rews = brak_rews[None].expand(self.num_plan, *brak_rews.shape)
        #stop_rews = stop_rews[None].expand(self.num_plan, *stop_rews.shape)
        #free = free[None].expand(self.num_plan, *free.shape)

        # If it is idle, make it LANE_FOLLOW
        cmd_value = cmd.value-1
        cmd_value = 3 if cmd_value < 0 else cmd_value

        #action_values, _ = BellmanUpdater.get_action(
        #    delta_locs, delta_yaws, next_spds,
        #    waypoint_rews[...,cmd_value], brak_rews, stop_rews, free,
        #    torch.zeros((self.num_plan,2)).float().to(BellmanUpdater._device),
        #    extract=(
        #        torch.tensor([[0.,0.]]),  # location
        #        torch.tensor([0.]),       # yaw
        #        torch.tensor([spd]),      # spd
        #    )
        #)
        #action_values = action_values.squeeze(0)

        #action = int(Categorical(logits=action_values/self.temperature).sample())
        # action = int(action_values.argmax())

        #steer, throt, brake = map(float, BellmanUpdater._actions[action])
        steer, throt, brake = self.run_step_image_policy(input_data, timestamp, cmd)
        
        if self.noise_collect:
            steer += self.noiser()

        #if len(self.vizs) > self.num_per_flush:
        #    self.flush_data()

        rgb = np.concatenate([wide_rgbs[0], narr_rgbs[0]], axis=1)
        spd = ego.get('spd')
        self.vizs.append(visualize_obs(rgb, yaw/180*math.pi, (steer, throt, brake), spd, cmd=cmd.value, lbl=lbl_copy))

        if col:
            self.cleanup()
            raise Exception('Collector has collided!! Heading out :P')

        if spd < STOP_THRESH:
            self.stop_count += 1
        else:
            self.stop_count = 0
        
        if cmd_value in [4,5]:
            actual_steer = steer
        else:
            actual_steer = steer * 1.2

        self.prev_steer = actual_steer

        # Save data
        if self.num_frames % (self.num_repeat+1) == 0 and self.stop_count < MAX_STOP:
            # Note: basically, this is like fast-forwarding. should be okay tho as spd is 0.
            self.wide_rgbs.append(wide_rgbs)
            self.narr_rgbs.append(narr_rgbs)
            self.wide_sems.append(wide_sems)
            self.narr_sems.append(narr_sems)
            self.lbls.append(lbl)
            self.locs.append(loc)
            self.rots.append(yaw)
            self.spds.append(spd)
            self.cmds.append(cmd_value)

            inf = -1
            infs = CarlaDataProvider.get_infraction_list()
            if self.num_infractions < len(infs):
                inf_obj = infs[-1]
                inf = inf_obj.get_type().value
                self.num_infractions = len(infs)

            self.infs.append(inf)
            self.flush_data(self.num_samples)
            self.num_samples += 1
        
        self.num_frames += 1
        
        return carla.VehicleControl(steer=actual_steer, throttle=throt, brake=brake)

    def run_step_image_policy(self, input_data, timestamp, cmd):
        _, wide_rgb = input_data.get(f'Wide_RGB_0')
        _, narr_rgb = input_data.get(f'Narrow_RGB_0')

        # Crop images
        _wide_rgb = wide_rgb[self.wide_crop_top:,:,:3]
        _narr_rgb = narr_rgb[:-self.narr_crop_bottom,:,:3]

        _wide_rgb = _wide_rgb[...,::-1].copy()
        _narr_rgb = _narr_rgb[...,::-1].copy()

        _, ego = input_data.get('EGO')
        _, gps = input_data.get('GPS')


        #_, _, cmd = self.waypointer.tick(gps)

        spd = ego.get('spd')
        
        cmd_value = cmd.value-1
        cmd_value = 3 if cmd_value < 0 else cmd_value

        if cmd_value in [4,5]:
            if self.lane_changed is not None and cmd_value != self.lane_changed:
                self.lane_change_counter = 0

            self.lane_change_counter += 1
            self.lane_changed = cmd_value if self.lane_change_counter > {4:200,5:200}.get(cmd_value) else None
        else:
            self.lane_change_counter = 0
            self.lane_changed = None

        if cmd_value == self.lane_changed:
            cmd_value = 3

        _wide_rgb = torch.tensor(_wide_rgb[None]).float().permute(0,3,1,2).to(self.device)
        _narr_rgb = torch.tensor(_narr_rgb[None]).float().permute(0,3,1,2).to(self.device)
        
        if self.all_speeds:
            steer_logits, throt_logits, brake_logits = self.image_model.policy(_wide_rgb, _narr_rgb, cmd_value)
            # Interpolate logits
            steer_logit = self._lerp(steer_logits, spd)
            throt_logit = self._lerp(throt_logits, spd)
            brake_logit = self._lerp(brake_logits, spd)
        else:
            steer_logit, throt_logit, brake_logit = self.image_model.policy(_wide_rgb, _narr_rgb, cmd_value, spd=torch.tensor([spd]).float().to(self.device))

        
        action_prob = self.action_prob(steer_logit, throt_logit, brake_logit)

        brake_prob = float(action_prob[-1])

        steer = float(self.steers @ torch.softmax(steer_logit, dim=0))
        throt = float(self.throts @ torch.softmax(throt_logit, dim=0))

        steer, throt, brake = self.post_process(steer, throt, brake_prob, spd, cmd_value)


        steer, throt, brake = (2*np.random.random()-1, 0.75, 0)
        return steer, throt, brake
        #return carla.VehicleControl(steer=steer, throttle=throt, brake=brake)

    def post_process(self, steer, throt, brake_prob, spd, cmd):
        
        if brake_prob > 0.5:
            steer, throt, brake = 0, 0, 1
        else:
            brake = 0
            throt = max(0.4, throt)

        # # To compensate for non-linearity of throttle<->acceleration
        # if throt > 0.1 and throt < 0.4:
        #     throt = 0.4
        # elif throt < 0.1 and brake_prob > 0.3:
        #     brake = 1

        if spd > {0:10,1:10}.get(cmd, 20)/3.6: # 10 km/h for turning, 15km/h elsewhere
            throt = 0

        # if cmd == 2:
        #     steer = min(max(steer, -0.2), 0.2)

        # if cmd in [4,5]:
        #     steer = min(max(steer, -0.4), 0.4) # no crazy steerings when lane changing


        return steer, throt, brake
    
    def _lerp(self, v, x):
        D = v.shape[0]

        min_val = self.min_speeds
        max_val = self.max_speeds

        x = (x - min_val)/(max_val - min_val)*(D-1)

        x0, x1 = max(min(math.floor(x), D-1),0), max(min(math.ceil(x), D-1),0)
        w = x - x0

        return (1-w) * v[x0] + w * v[x1]

    def action_prob(self, steer_logit, throt_logit, brake_logit):

        steer_logit = steer_logit.repeat(self.num_throts)
        throt_logit = throt_logit.repeat_interleave(self.num_steers)

        action_logit = torch.cat([steer_logit, throt_logit, brake_logit[None]])

        return torch.softmax(action_logit, dim=0)


def _random_string(length=10):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))
