#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Common libs
import signal
import os

# Dataset
from datasets.Vaihingen_subregion import *
from torch.utils.data import DataLoader
import nvidia_smi
from utils.config import Config
from utils.trainer_mprm import ModelTrainer

from models.architectures import * 
from time import gmtime, strftime
import time

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

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 6 # 6 #6 #1 # 10

 

    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'nearest_upsample',
                    'nearest_upsample']

    ###################
    # KPConv parameters
    ###################
    
    # Radius of the input sphere
    in_radius = 24 #6 #24 #24 #24 #36 #6 #9 #6.0 #1.5  ???
    sub_radius = 6 #6 #12 # 6 #12 #12
    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.24 

    # Number of kernel points
    num_kernel_points = 15

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5 

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 1 
    
    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.0 

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 64
    in_features_dim = 4 #4 #1 #take height as a feature

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
    repulse_extent = 1.2                     # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Number of batch
    batch_num = 2 

    max_epoch = 80
    learning_rate = 0.01 
    momentum = 0.98
    lr_decays = {}
    lr_decays = {i: 0.98 for i in range(1, 1000)}
    grad_clip_norm = 1
    epoch_steps = 600
    validation_size = 50
    checkpoint_gap = 5 

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, True, False] 
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.04

    segloss_balance = 'none' #'none'
    class_w=[1, 1, 1, 1, 1, 1, 1, 1, 1]
    xyz_offset = [0,0, 0] 

    saving = True
    saving_path = None

    edge_limit = 80
    agg_method = 'mean_mean'
    previous_training_path = '' 
    chkp_idx = -1 
    model_name = 'KPFCNN_mprm' 
    loss_type = 'region_mprm_loss_any'  
    contrast_start = 800
    contrast_thd = 0.4
    heads = '' 
    anchor_method = 'reduce1'
    grad_method = 'method2'
    optopt = 'sgd'
    use_pretrained= False# True
    pretxt= ''#
    use_dropout = 0
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
        
        if free>8:
            break
        
        print('Waiting...')
        time.sleep(1200)

    ###############
    # Previous chkp
    ###############
    config = VaiConfig()

    previous_training_path = ''
    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = config.chkp_idx ##None
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('vai_results', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('vai_results', previous_training_path, 'checkpoints', chosen_chkp)

    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    if previous_training_path:
        config.load(os.path.join('vai_results', previous_training_path))
        config.saving_path = None

    if len(sys.argv) > 1:
        config.saving_path = sys.argv[1]

    # Initialize datasets
    training_dataset = VaiDataset(config, set='training', use_potentials=True)
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
    if config.model_name == 'KPFCNN_mprm':
        net = KPFCNN_mprm(config, training_dataset.label_values, training_dataset.ignored_labels)
    elif config.model_name == 'KPFCNN_mprm_ele':
        net = KPFCNN_mprm_ele(config, training_dataset.label_values, training_dataset.ignored_labels) 
    
    print(config.model_name, int(count_parameters(net)//1000))

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
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp, use_pretrained=config.use_pretrained, pretxt=config.pretxt)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training
    trainer.train(net, training_loader, test_loader, config)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)
