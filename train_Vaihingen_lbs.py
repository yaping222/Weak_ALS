#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

# Common libs
import signal
import os

# Dataset
from datasets.Vaihingen_lbs import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPFCNN
import argparse
import nvidia_smi
from os.path import join


parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-model_n", type=str, default='')    
parser.add_argument(
    "-ck", type=str, default='')    
parser.add_argument(
    "-previous_training_path", type=str, default='')    
parser.add_argument(
    "-chkp_idx", type=int, default=None)  
parser.add_argument(
    "-contrast_start", type=int, default=0)  
parser.add_argument(
    "-contrast_thd", type=float, default=0.2) 
parser.add_argument(
    "-cl_method", type=str, default='') 
parser.add_argument(
    "-dropout", type=float, default=None) 
# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class VaiConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'Vaihingen'

    # Number of classes in the dataset (This value is overwritten by datasechkp_0105_4t class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 6 # 10

    #########################
    # Architecture definition
    #########################

    # Define layers
    # according to the paper, rigid convs perform better on semantic3D
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    ###################
    # KPConv parameters
    ###################
    # batch_number vs epoch_steps
    ff=0.5
    
    # Radius of the input sphere
    in_radius = 24 #36 #6 #9 #6.0 #1.5  ???
    
    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.24 #0.36

    # Number of kernel points
    num_kernel_points = 15

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5 #2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 6 #6.0 #5.0 is according to paper

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.0 #1.2 #1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 64
    in_features_dim = 4 #take height as a feature

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02 # 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2 #*10                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    max_epoch = 30 
    learning_rate = 0.001 
    momentum = 0.98
    lr_decays = {}
    for i in range(1,100):
        if i%5==0 and i<101:        
            lr_decays[i]=0.7 
        else:
            lr_decays[i]=1 

    grad_clip_norm = 100.0
    batch_num = 4 
    epoch_steps = 2000
    validation_size = 200
    checkpoint_gap = 5 

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, True, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001

    segloss_balance = 'none' #'none'
    class_w=[1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    saving = True
    saving_path = None
    ck = ''
    model_n = ''
    contrast_start = 0
    contrast_thd = 0.2
    cl_method = 'weighted' # weighted
    dropout = 0.5 #None
# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    for i in range(10000):
        
        nvidia_smi.nvmlInit()
        
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        free = info.free//(1024*1024*1024)
        print("Free memory:", free)
        
        if free>9:
            break
        
        print('Waiting...')
        time.sleep(1200)
    ###############
    # Previous chkp
    ###############
    args = parser.parse_args()
    previous_training_path = args.previous_training_path
    print(previous_training_path)
    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = args.chkp_idx
    print('chkp_idx: ', chkp_idx)
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('vai_pseudo_results', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('vai_pseudo_results', previous_training_path, 'checkpoints', chosen_chkp)
        
    else:
        chosen_chkp = None
    print('chosen_chkp: ', chosen_chkp)
    # exit()
    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = VaiConfig()
    if previous_training_path:
        config.load(os.path.join('vai_pseudo_results', previous_training_path))
        config.saving_path = None


    config.update_arg(args)
    print(config.model_n, config.ck)
    
    w_n = 'Vaihingen3D_Train_'+config.ck+'_weight.txt'
    w_file = join('.../test', config.model_n, w_n)
    print(w_file)
    if exists(w_file):
        config.class_w = np.genfromtxt(w_file, delimiter=' ')
        print('config.class_w: ', config.class_w)

    # Initialize datasets
    training_dataset = VaiDataset(config, set='training', use_potentials=True)
#    training_dataset = VaiDataset(config, set='training', use_potentials=False)
    test_dataset = VaiDataset(config, set='validation', use_potentials=True)

    # Initialize samplers
    training_sampler = VaiSampler(training_dataset)
    test_sampler = VaiSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=VaiCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=VaiCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)
    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = KPFCNN(config, training_dataset.label_values, training_dataset.ignored_labels)
    # print(config.model_n, int(count_parameters(net)//1000))
    debug = False
    if debug:
        print('\n*************************************\n')
        print(net)
        print('\n*************************************\n')
        for param in net.parameters():
            if param.requires_grad:
                print(param.shape)
        print('\n*************************************\n')
        print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
        print('\n*************************************\n')

    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training
    trainer.train(net, training_loader, test_loader, config)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)

