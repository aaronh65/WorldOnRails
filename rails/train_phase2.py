from copy import deepcopy
import tqdm
import numpy as np
import torch
from .rails import RAILS
from .datasets import data_loader
from .logger import Logger

def main(args):
    
    rails = RAILS(args)
    train_data = data_loader('main', args, mode='train', priority=True)
    val_data = data_loader('main', args, mode='val', priority=True)
    logger = Logger('carla_train_phase2', args)
    save_dir = logger.save_dir
    
    if args.resume:
        print ("Loading checkpoint from", args.resume)
        if rails.multi_gpu:
            rails.main_model.module.load_state_dict(torch.load(args.resume))
        else:
            rails.main_model.load_state_dict(torch.load(args.resume))
        start = int(args.resume.split('main_model_')[-1].split('.th')[0])
    else:
        start = 0

    global_it = 0
    for epoch in range(start,start+args.num_epoch):

        # train
        for wide_rgbs, wide_sems, narr_rgbs, narr_sems, act_vals, spds, cmds, infs in tqdm.tqdm(train_data, desc='Epoch {}'.format(epoch)):
            opt_info = rails.train_main(wide_rgbs, wide_sems, narr_rgbs, narr_sems, act_vals, spds, cmds, infs)
            opt_info['epoch'] = epoch
            
            if global_it % args.num_per_log == 0:
                logger.log_main_info(global_it, opt_info, mode='train')
        
            global_it += 1

        # val
        rails.main_model.eval()

        val_info = {'epoch': epoch, 'val_seg_loss': [], 'val_act_loss': [], 'val_loss': [], 'val_exp_loss': []}
        for wide_rgbs, wide_sems, narr_rgbs, narr_sems, act_vals, spds, cmds, infs in tqdm.tqdm(val_data, desc='Epoch {}'.format(epoch)):
            opt_info = rails.val_main(wide_rgbs, wide_sems, narr_rgbs, narr_sems, act_vals, spds, cmds, infs)
            val_info['val_seg_loss'].append(opt_info['seg_loss'])
            val_info['val_act_loss'].append(opt_info['act_loss'])
            val_info['val_exp_loss'].append(opt_info['exp_loss'])
            val_info['val_loss'].append(opt_info['loss'])
        val_info['val_seg_loss'] = np.mean(val_info['val_seg_loss'])
        val_info['val_act_loss'] = np.mean(val_info['val_act_loss'])
        val_info['val_exp_loss'] = np.mean(val_info['val_exp_loss'])
        val_info['val_loss'] = np.mean(val_info['val_loss'])
        opt_info.pop('seg_loss')
        opt_info.pop('act_loss')
        opt_info.pop('exp_loss')
        opt_info.pop('loss')
        val_info.update(opt_info)
        logger.log_main_info(global_it, val_info, mode='val')

        rails.main_model.train()
    
        # Save model
        if (epoch+1) % args.num_per_save == 0:
            save_path = f'{save_dir}/main_model_{epoch+1}.th'
            torch.save(rails.main_model_state_dict(), save_path)
            print (f'saved to {save_path}')

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    
    #parser.add_argument('--resume', default=None)
    parser.add_argument('--resume', default='/data/aaronhua/wor/training/main/dian/main_model_10.th')
    
    parser.add_argument('--data-dir', default='/ssd1/aaronhua/wor/data/main/train_stream')
    parser.add_argument('--config-path', default='config.yaml')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    
    # Training data config
    parser.add_argument('--fps', type=float, default=20)
    parser.add_argument('--num-repeat', type=int, default=4)    # Should be consistent with autoagents/collector_agents/config.yaml

    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0)
    
    parser.add_argument('--num-per-log', type=int, default=250, help='per iter')
    parser.add_argument('--num-per-save', type=int, default=2, help='per epoch')
    
    parser.add_argument('--balanced-cmd', action='store_true')

    args = parser.parse_args()
    main(args)
