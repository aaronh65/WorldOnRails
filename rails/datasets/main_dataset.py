import ray
import cv2
import glob
import yaml
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from common.augmenter import augment
from pathlib import Path
from utils import filter_sem

# CHANNELS = [
#     4,  # Pedestrians    
#     6,  # Road lines
#     7,  # Road masks
#     8,  # Side walks
#     10, # Vehicles
#     # 12, # Traffic light poles
#     # 18, # Traffic boxes
# ]


split_dict = {'train': ['training', 'devtest'], 'val': ['testing']}

class VisualizationDataset(Dataset):
    
    def __init__(self, data_dir, config_path):
        super().__init__()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.T = config['num_plan']
        self.seg_channels = config['seg_channels']
        
        self.num_speeds = config['num_speeds']
        self.num_steers = config['num_steers']
        self.num_throts = config['num_throts']

        self.num_frames = 0
        self.txn_map = dict()
        self.idx_map = dict()
        self.file_map = dict()

        # Load dataset
        for full_path in glob.glob(f'{data_dir}/**'):
            txn = lmdb.open(
                full_path,
                max_readers=1, readonly=True,
                lock=False, readahead=False, meminit=False).begin(write=False)
            
            n = int(txn.get('len'.encode()))
            if n < self.T+1:
                print (full_path, 'is too small. consider deleting it.')
                txn.__exit__()
            else:
                offset = self.num_frames
                for i in range(n-self.T):
                    self.num_frames += 1
                    self.txn_map[offset+i] = txn
                    self.idx_map[offset+i] = i
                    self.file_map[offset+i] = full_path

        print(f'{data_dir}: {self.num_frames} frames')
    
    def __len__(self):
        return self.num_frames
        
    def __getitem__(self, idx):

        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]

        locs = self.__class__.access('loc', lmdb_txn, index, self.T+1, dtype=np.float32)
        rots = self.__class__.access('rot', lmdb_txn, index, self.T, dtype=np.float32)
        spds = self.__class__.access('spd', lmdb_txn, index, self.T, dtype=np.float32).flatten()
        lbls = self.__class__.access('lbl', lmdb_txn, index+1, self.T, dtype=np.uint8).reshape(-1,96,96,12)
        maps = self.__class__.access('map', lmdb_txn, index+1, self.T, dtype=np.uint8).reshape(-1,1536,1536,12)

        rgb = self.__class__.access('rgb',  lmdb_txn, index, 1, dtype=np.uint8).reshape(720,1280,3)
        cmd = self.__class__.access('cmd', lmdb_txn, index, 1, dtype=np.float32).flatten()
        act = self.__class__.access('act', lmdb_txn, index, 1, dtype=np.float32).flatten()

        # Crop cameras
        rgb = rgb[:,:,::-1]

        return rgb, maps, lbls, locs, rots, spds, int(cmd), act

    @staticmethod
    def access(tag, lmdb_txn, index, T, dtype=np.float32):
        return np.stack([np.frombuffer(lmdb_txn.get((f'{tag}_{t:05d}').encode()), dtype) for t in range(index,index+T)])


class MainDataset(Dataset):
    def __init__(self, data_dir, config_path, priority=False, mode=''):
        super().__init__()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.T = config['num_plan']
        self.camera_yaws = config['camera_yaws']
        self.wide_crop_top = config['wide_crop_top']
        self.narr_crop_bottom = config['narr_crop_bottom']
        self.seg_channels = config['seg_channels']
        
        self.num_speeds = config['num_speeds']
        self.num_steers = config['num_steers']
        self.num_throts = config['num_throts']

        # Ablation options
        self.multi_cam = config['multi_cam']

        self.num_frames = 0
        self.txn_map = dict()
        self.idx_map = dict()
        self.yaw_map = dict()
        self.file_map = dict()

        # Load dataset
        data_dir = Path(data_dir) / 'data'
        routes = list(sorted(data_dir.glob('*')))
        folders = []
        for route in routes:
            folders.extend(list(sorted(route.glob('*'))))

        bad_paths = ''
        ddict = {'hard':[], 'all':[]}
        self.hard_frames = 0
        for i, full_path in enumerate(folders):
            full_path = str(full_path)

            # check for splits
            if mode != '':
                add = False
                for tag in split_dict[mode]:
                    if tag in full_path:
                        add = True
                        break
                if not add:
                    continue

            try:
                txn = lmdb.open(
                    full_path,
                    max_readers=1, readonly=True,
                    lock=False, readahead=False, meminit=False).begin(write=False)

                n = int(txn.get('len'.encode()))
                if n < self.T+1:
                    print (full_path, ' is too small. consider deleting it.')
                    txn.__exit__()
                else:
                    offset = self.num_frames
                    for i in range(n-self.T):
                        self.num_frames += 1
                        for j in range(len(self.camera_yaws)):
                            self.txn_map[(offset+i)*len(self.camera_yaws)+j] = txn
                            self.idx_map[(offset+i)*len(self.camera_yaws)+j] = i
                            self.yaw_map[(offset+i)*len(self.camera_yaws)+j] = j
                            self.file_map[(offset+i)*len(self.camera_yaws)+j] = full_path
                            ddict['all'].append((offset+i)*len(self.camera_yaws)+j)

                            infs = self.__class__.access('inf', txn, i, self.T+1, dtype=np.float32)
                            # ignore stop sign infractions
                            if len(set(infs.flatten())) > 1 and 10 not in set(infs.flatten()): 
                                ddict['hard'].append((offset+i)*len(self.camera_yaws)+j)
                                self.hard_frames += 1

            except Exception as e:
                print(e)
                bad_paths += f'{str(full_path)}\n'

                    
        #print(ddict['hard'])
        #print(ddict['all'])

        print(f'{data_dir}: {self.num_frames} frames (x{len(self.camera_yaws)})')
        print('the following directories had errors:')
        print(bad_paths)

        self.ddict = ddict
        self.hard_frames /= len(self.camera_yaws)

        if self.hard_frames == 0:
            print(f'WARNING: no hard samples available for {data_dir} in {mode} mode')
            self.ddict['hard'] = self.ddict['all']
        else:
            print(f'{data_dir}: {self.hard_frames} hard frames (x{len(self.camera_yaws)})')

        if self.multi_cam:
            self.dataset_len = self.num_frames*len(self.camera_yaws)
        else:
            self.dataset_len = self.num_frames*len(self.camera_yaws)

        self.priority = priority
        self.hard_prop = config['hard_prop']
        #self.update_rate = self.dataset_len / config['num_workers']

        
    def __len__(self):
        return self.dataset_len
                
    def __getitem__(self, idx):
        
        if not self.multi_cam:
            idx *= len(self.camera_yaws)

        
        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]
        cam_index = self.yaw_map[idx]
        
        locs = self.__class__.access('loc', lmdb_txn, index, self.T+1, dtype=np.float32)
        rots = self.__class__.access('rot', lmdb_txn, index, self.T, dtype=np.float32)
        spds = self.__class__.access('spd', lmdb_txn, index, self.T, dtype=np.float32).flatten()
        lbls = self.__class__.access('lbl', lmdb_txn, index+1, self.T, dtype=np.uint8).reshape(-1,96,96,12)

        wide_rgb = self.__class__.access('wide_rgb_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,480,3)
        wide_sem = self.__class__.access('wide_sem_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,480)
        narr_rgb = self.__class__.access('narr_rgb_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,384,3)
        cmd = self.__class__.access('cmd', lmdb_txn, index, 1, dtype=np.float32).flatten()
        
        wide_sem = filter_sem(wide_sem, self.seg_channels)

        # Crop cameras
        wide_rgb = wide_rgb[self.wide_crop_top:,:,::-1]
        wide_sem = wide_sem[self.wide_crop_top:]
        narr_rgb = narr_rgb[:-self.narr_crop_bottom,:,::-1]

        # EXPERIMENTAL
        infs = self.__class__.access('inf', lmdb_txn, index, self.T+1, dtype=np.float32)

        # EXPERIMENTAL
        #infs = self.__class__.access('inf', lmdb_txn, 0, self.num_frames, dtype=np.float32)
        #print(infs)

        return wide_rgb, wide_sem, narr_rgb, lbls, locs, rots, spds, int(cmd), infs


    @staticmethod
    def access(tag, lmdb_txn, index, T, dtype=np.float32):
        return np.stack([np.frombuffer(lmdb_txn.get((f'{tag}_{t:05d}').encode()), dtype) for t in range(index,index+T)])


class LabeledMainDataset(MainDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmenter = augment(0.5)
        
    def __getitem__(self, idx):

        if self.priority:
            if np.random.random() < self.hard_prop:
                idx = np.random.choice(self.ddict['hard'])
            else:
                idx = np.random.choice(self.ddict['all'])

        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]
        cam_index = self.yaw_map[idx]

        wide_rgb = self.__class__.access('wide_rgb_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,480,3)
        wide_sem = self.__class__.access('wide_sem_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,480)
        narr_rgb = self.__class__.access('narr_rgb_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,384,3)
        narr_sem = self.__class__.access('narr_sem_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,384,4)[...,2]
        cmd = self.__class__.access('cmd', lmdb_txn, index, 1, dtype=np.float32).flatten()
        spd = self.__class__.access('spd', lmdb_txn, index, 1, dtype=np.float32).flatten()

        act_val = self.__class__.access('act_val_{}'.format(cam_index), lmdb_txn, index, 1, dtype=np.float32).reshape(6,self.num_steers*self.num_throts+1,self.num_speeds)

        wide_sem = filter_sem(wide_sem, self.seg_channels)
        narr_sem = filter_sem(narr_sem, self.seg_channels)

        # Crop cameras
        wide_rgb = wide_rgb[self.wide_crop_top:,:,::-1]
        wide_sem = wide_sem[self.wide_crop_top:]
        narr_rgb = narr_rgb[:-self.narr_crop_bottom,:,::-1]
        narr_sem = narr_sem[:-self.narr_crop_bottom]

        # Augment
        wide_rgb = self.augmenter(images=wide_rgb[None])[0]
        narr_rgb = self.augmenter(images=narr_rgb[None])[0]

        infs = self.__class__.access('inf', lmdb_txn, index, self.T+1, dtype=np.float32)

        return wide_rgb, wide_sem, narr_rgb, narr_sem, act_val, float(spd), int(cmd), infs


@ray.remote
class RemoteMainDataset(MainDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_vals = dict()

    def num_frames(self):
        return self.num_frames

    def save(self, base_idx, action_values):
        """
        Cache the action values
        """
        for i in range(len(self.camera_yaws)):
            idx = base_idx + i
            self.act_vals[idx] = action_values[i]

    def commit(self):
        """
        Commit the saved cache
        """

        for idx, act_val in self.act_vals.items():

            file_name = self.file_map[idx]
            file_idx  = self.idx_map[idx]
            yaw_idx   = self.yaw_map[idx]

            lmdb_env = lmdb.open(file_name, map_size=int(1e13))
            with lmdb_env.begin(write=True) as txn:
                txn.put(
                    f'act_val_{yaw_idx}_{file_idx:05d}'.encode(),
                    np.ascontiguousarray(act_val).astype(np.float32),
                )

            lmdb_env.close()
            print ('wrote to {}'.format(file_name))


if __name__ == '__main__':
    
    #dataset = MainDataset('/ssd2/dian/challenge_data/main_trajs_nocrash_nonoise', '/home/dianchen/carla_challenge/experiments/config_nocrash.yaml')
    #dataset = MainDataset('/data3/aaronhua/wor/data/main/train_stream', '/home/aaronhua/WorldOnRails/config.yaml', mode='train')
    #dataset = MainDataset('/data3/aaronhua/wor/data/main/val_stream', '/home/aaronhua/WorldOnRails/config.yaml', mode='val')
    #dataset = MainDataset('/data3/aaronhua/wor/data/main/devtest_stream', '/home/aaronhua/WorldOnRails/config.yaml', mode='train')
    dataset = MainDataset('/ssd0/aaronhua/wor/data/main/train_stream', '/home/aaronhua/WorldOnRails/config.yaml', mode='train')
    
    #for i, data in enumerate(dataset):
    #    if i % 3 != 0 :
    #        continue
    #    wide_rgb, wide_sem, narr_rgb, lbls, locs, rots, spds, cmd, infs = data
    #    #cv2.imshow('rgb', wide_rgb)
    #    #cv2.waitKey(100)
    #    #print(wide_rgb.shape)
    #    for inf in infs.flatten():
    #        if inf != -1:
    #            print(i, infs)
    #            continue
