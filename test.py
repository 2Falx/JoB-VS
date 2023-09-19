# -*- coding: utf-8 -*-
from bdb import Breakpoint
import os
import time
import argparse
import numpy as np

import libs.trainer as trainer

from libs.model.model import ROG
from settings import plan_experiment
from libs.dataloader import dataloader

import torch
import torch.optim as optim
import torch.cuda.amp as amp
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

tasks = {'0': 'Vessel_Segmentation'}


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1234' + port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, args):
    print(f"Running on rank {rank}.")
    setup(rank, world_size, args.port)
        
    info, model_params = plan_experiment(
        tasks[args.task], args.batch,
        args.patience,
        rank, args.model, args.data_ver)

    # PATHS AND DIRS
    args.save_path = os.path.join(info['output_folder'], args.name)
    images_path = os.path.join(args.save_path, 'volumes')
    
    images_path = os.path.join(images_path)

    
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(os.path.join(images_path, 'brain'), exist_ok=True)
    os.makedirs(os.path.join(images_path, 'vessels'), exist_ok=True)
    os.makedirs(os.path.join(images_path, 'vessels_brain'), exist_ok=True)
    os.makedirs(os.path.join(images_path, 'vessels_logits'), exist_ok=True)
    os.makedirs(os.path.join(images_path, 'vessels_brain_logits'), exist_ok=True)

    # SEEDS
    np.random.seed(info['seed'])
    torch.manual_seed(info['seed'])

    cudnn.deterministic = False  # Normally is False
    cudnn.benchmark = args.benchmark  # Normaly is True

    # CREATE THE NETWORK ARCHITECTURE
    if args.model == 'ROG':
        model = ROG(model_params).to(rank)
    
    ddp_model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        f = open(os.path.join(args.save_path, 'architecture.txt'), 'w')
        print(model, file=f)
        
    # CHECKPOINT
    if args.load_weights is not None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(args.load_weights, map_location=map_location)
        
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        
        if 'rng' in checkpoint.keys():
            np.random.set_state(checkpoint['rng'][0])
            torch.set_rng_state(checkpoint['rng'][1])
        
        if 'module.' not in list(checkpoint.keys())[0]:
            checkpoint = {'module.' + k: v for k, v in checkpoint.items()}
        
        model_dict = ddp_model.state_dict()
        # Match pre-trained weights that have same shape as current model.
        
        pre_train_dict_match = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
        
        # Weights that do not have match from the pre-trained model.
        not_load_layers = [k for k in model_dict.keys() if k not in pre_train_dict_match.keys()]
        
        # Log weights that are not loaded with the pre-trained weights.
        if not_load_layers and rank==0:
            for k in not_load_layers:
                print("Network weights {} not loaded.".format(k))
        
        # Load pre-trained weights.
        ddp_model.load_state_dict(pre_train_dict_match, strict=False)
        
    # DATASETS    
    test_dataset = dataloader.Medical_data(False, info['data_file'], info['root'], info['val_size'], val=True)

    print(f"Info in_size {info['in_size']}") # 192 192 96

    # DATALOADERS
    test_loader = DataLoader(
        test_dataset, sampler=None, shuffle=False, batch_size=1, num_workers=0)

    # TRAIN THE MODEL
    torch.cuda.empty_cache()

    # Loading the best model for testing
    dist.barrier()
    torch.cuda.empty_cache()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    name = args.load_model + '.pth.tar'
    
    checkpoint = torch.load(os.path.join(args.save_path, name), map_location=map_location)
    
    torch.set_rng_state(checkpoint['rng'][1]) 
    
    ddp_model.load_state_dict(checkpoint['state_dict'])
    
    if rank == 0:
        print(f'Testing epoch with best dice ({checkpoint["epoch"]}: dice {checkpoint["dice"]})')

    # EVALUATE THE MODEL
    trainer.test(info, ddp_model, test_loader, images_path,
                 info['data_file'], rank, world_size)
    
    dist.barrier()
    cleanup()


if __name__ == '__main__':
    # SET THE PARAMETERS
    parser = argparse.ArgumentParser()
    # EXPERIMENT DETAILS
    parser.add_argument('--task', type=str, default='0',
                        help='Task to train/evaluate (default: 4)')
    parser.add_argument('--model', type=str, default='ROG',
                        help='Model to train with the ROG training curriculum (default: ROG)')
    parser.add_argument('--data_ver', type=str, default='your/saving/processed/data/path',
                        help='Path to data')
    parser.add_argument('--name', type=str, default='JoB-VS',
                        help='Name of the current experiment (default: ROG)')
    parser.add_argument('--AT', action='store_true', default=False,
                        help='Train a model with Free AT')
    parser.add_argument('--fold', type=str, default=1,
                        help='Which fold to run. Value from 1 to 2')
    parser.add_argument('--test', action='store_false', default=True,
                        help='Evaluate a model')
    parser.add_argument('--aux_train', action='store_true', default=False,
                        help='Sample more patches per patient')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Continue training a model')
    parser.add_argument('--ft', action='store_true', default=False,
                        help='Fine-tune a model (will not load the optimizer)')
    parser.add_argument('--load_model', type=str, default='best_dice',
                        help='Weights to load (default: best_dice)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Name of the folder with the pretrained model')
    parser.add_argument('--ixi', action='store_true', default=False,
                        help='Evaluate a model in IXI dataset')
    
    # TRAINING HYPERPARAMETERS
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Maximum number of epochs (default: 1000)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience of the scheduler (default: 50)')
    parser.add_argument('--batch', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--load_weights', type=str, default=None,
                        help='Path to load initial weights (default: None)')
    
    # ADVERSARIAL TRAINING AND TESTING
    parser.add_argument('--eps', type=float, default=8.,
                        help='Epsilon for the adv. attack (default: 8/255)')
    parser.add_argument('--alpha_vessels', type=float, default=0.5,
                        help='Multiplication factor in vessels loss'),
    parser.add_argument('--alpha_brain', type=float, default=0.5,
                        help='Multiplication factor in brain loss'),
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU(s) to use (default: 0)')
    parser.add_argument('--port', type=str, default='5')
    parser.add_argument('--benchmark', action='store_false', default=True,
                        help='Deactivate CUDNN benchmark')
    parser.add_argument('--mask', action='store_true', default=False,
                        help='Use data with the brain masks')
    parser.add_argument('--cldice', action='store_true', default=False,
                        help='Use cldice for vessel segmentation')
    parser.add_argument('--detection', action='store_true', default=False,
                        help='Evaluate with detection metrics')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold to evaluate the predictions (default: None)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    world_size = torch.cuda.device_count()
    
    if world_size > 1:
        mp.spawn(main, args=(world_size, args,), nprocs=world_size, join=True)
    else:
        # To allow breakpoints
        main(0, 1, args)
