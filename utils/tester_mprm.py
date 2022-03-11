#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 15:32:00 2021

@author: yp
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:42:12 2021
@author: yp
"""

#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the test of any model
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import torch
import torch.nn as nn
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
import json
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply, remove_empty

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from sklearn.metrics import confusion_matrix
import pickle
#from utils.visualizer import show_ModelNet_models

# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#


class ModelTester:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, chkp_path=None, on_gpu=True):

        ############
        # Parameters
        ############

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)
        self.ck_name=chkp_path.split('/')[-1].split('.tar')[0]
        ##########################
        # Load previous checkpoint
        ##########################

        checkpoint = torch.load(chkp_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        net.eval()
        print("Model and training state restored.")

        return

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def cloud_segmentation_test(self, net, test_loader, config, num_votes=100, debug=False):
        """
        Test method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.95
        test_radius_ratio = 0.7
        softmax = torch.nn.Softmax(1)
        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate global prediction over test clouds
        self.test_probs = [np.zeros((l.shape[0], nc_model)) for l in test_loader.dataset.input_labels]

        # Test saving path
        if config.saving:
            if 'DFC' in config.dataset=='DFC':
                test_path = join('test/DFC', config.saving_path.split('/')[-1])
            else:
                test_path = join('test', config.saving_path.split('/')[-1])
            if not exists(join(test_path, self.ck_name)):
                makedirs(join(test_path, self.ck_name))
            if not exists(join(test_path, self.ck_name, 'predictions')):
                makedirs(join(test_path, self.ck_name, 'predictions'))
            if not exists(join(test_path,self.ck_name, 'probs')):
                makedirs(join(test_path, self.ck_name, 'probs'))
            if not exists(join(test_path, self.ck_name, 'potentials')):
                makedirs(join(test_path, self.ck_name, 'potentials'))
        else:
            test_path = None

        # If on validation directly compute score
        if test_loader.dataset.set == 'validation':
            val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in test_loader.dataset.label_values:
                if label_value not in test_loader.dataset.ignored_labels:
                    if config.dataset=='rot':
                        val_proportions[i] = np.sum([np.sum(np.array(list(map(self.subs.get, labels))).astype('int8') == label_value)
                                                      for labels in test_loader.dataset.validation_labels])
                    else:
                        val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                      for labels in test_loader.dataset.validation_labels])
                        # print(test_loader.dataset.validation_labels[0])
                    i += 1
        else:
            val_proportions = None

        #####################
        # Network predictions
        #####################

        test_epoch = 0
        last_min = -0.5

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print('Initialize workers')
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                self.logits, self.class_logits, self.cam = net(batch, config)

                t += [time.time()]

                # Get probs and labels
                stacked_probs = softmax(self.logits).cpu().detach().numpy()
                s_points = batch.points[0].cpu().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                in_inds = batch.input_inds.cpu().numpy()
                cloud_inds = batch.cloud_inds.cpu().numpy()
                torch.cuda.synchronize(self.device)

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    points = s_points[i0:i0 + length]
                    probs = stacked_probs[i0:i0 + length]
                    inds = in_inds[i0:i0 + length]
                    c_i = cloud_inds[b_i]

                    if 0 < test_radius_ratio < 1:
                        mask = np.sum(points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2
                        inds = inds[mask]
                        probs = probs[mask]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                    i0 += length

                # Average timing
                t += [time.time()]
                if i < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})'
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2])))

            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.min_potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))
            #print([np.mean(pots) for pots in test_loader.dataset.potentials])

            # Save predicted cloud
            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                # Show vote results (On subcloud so it is not the good values here)
                if test_loader.dataset.set == 'validation':
                    print('\nConfusion on sub clouds')
                    Confs = []
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Insert false columns for ignored labels
                        probs = np.array(self.test_probs[i], copy=True)
                        print('probs2: ', np.sum(probs==0))
                        # print('probs: ', probs.shape)
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)
                        # Predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                        # print('preds_val: ', np.unique(preds))
                        # Targets
                        targets = test_loader.dataset.input_labels[i]
                        # print('targets_val: ', np.unique(targets))
                        # Confs
                        Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    print(s + '\n')

                # Save real IoU once in a while
                # if int(np.ceil(new_min)) % 10 == 0:
                all_pseudo_lbs=dict()
                all_pseduo_probs=dict()
                if last_min > num_votes:
                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    proj_probs = []
                    for i, file_path in enumerate(test_loader.dataset.files):
                        # points = #test_loader.dataset.load_evaluation_points(file_path)
                        print(i, file_path, test_loader.dataset.test_proj[i].shape, self.test_probs[i].shape)

                        print(test_loader.dataset.test_proj[i].dtype, np.max(test_loader.dataset.test_proj[i]))
                        print(test_loader.dataset.test_proj[i][:5])
                        # test_probs_i = remove_empty(self.test_probs[i], points)
                        # Reproject probs on the evaluations points
                        probs = self.test_probs[i][test_loader.dataset.test_proj[i], :]
                        print('probs: ', probs.shape)
                        proj_probs += [probs]
                        
                        fn = file_path.split('/')[-1].split('.txt')[0]
                        unlabel_pts = self.test_probs[i][:,0]==0
                        all_pseduo_probs[fn] = self.test_probs[i]
                        pseudo_lbs = np.argmax(self.test_probs[i], axis=1)+1
                        pseudo_lbs[unlabel_pts] = 0
                        all_pseudo_lbs[fn] = pseudo_lbs
                        
                    pl_path= join(test_path, self.ck_name+'_'+'pseudo.pickle')
                    prob_path= join(test_path, self.ck_name+'_'+'probs.pickle')
                    with open(pl_path, 'wb') as fff:
                        pickle.dump(all_pseudo_lbs, fff)
                    with open(prob_path, 'wb') as fff:
                        pickle.dump(all_pseduo_probs, fff)
                    
                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

                    # Show vote results
                    if test_loader.dataset.set == 'validation':
                        print('Confusion on full clouds')
                        t1 = time.time()
                        Confs = []
                        for i, file_path in enumerate(test_loader.dataset.files):
                            
                            # Insert false columns for ignored labels
                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    proj_probs[i] = np.insert(proj_probs[i], l_ind, 0, axis=1)
                          # Get the predicted labels
                            preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)
                            
                            # Confusion
                            targets = test_loader.dataset.validation_labels[i]                 
                                                        
                            Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                        # Regroup confusions
                        C = np.sum(np.stack(Confs), axis=0)
                        print('conf7: ', C)
                        c_path1= join(test_path, self.ck_name, 'predictions/confusion_o.txt')
                        np.savetxt(c_path1, C, delimiter=' ', fmt='%i')
                        # Remove ignored labels from confusions
                        for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                            if label_value in test_loader.dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)
                        # save confusion
                        c_path= join(test_path, self.ck_name, 'predictions/confusion.txt')
                        np.savetxt(c_path, C, delimiter=' ', fmt='%i')
                        
                        IoUs = IoU_from_confusions(C)
                        mIoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * mIoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        print('-' * len(s))
                        print(s)
                        print('-' * len(s) + '\n')

                    print('Saving clouds')
                    if config.dataset == 'rotterdam':
                        break
                    t1 = time.time()
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Get file
                        points = test_loader.dataset.load_evaluation_points(file_path)

                        # Get the predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                        # target
                        targets = test_loader.dataset.validation_labels[i]

                        error_map = preds!=targets
                        error_map = error_map.astype('int8')
                        
                        # Save plys
                        cloud_name = file_path.split('/')[-1]
                        test_name = join(test_path, self.ck_name, 'predictions', cloud_name)
                        write_ply(test_name,
                                  [points, preds, targets, error_map],
                                  ['x', 'y', 'z', 'preds', 'targets', 'error'])
                        test_name2 = join(test_path, self.ck_name, 'probs', cloud_name)
                        prob_names = ['_'.join(test_loader.dataset.label_to_names[label].split())
                                      for label in test_loader.dataset.label_values]
                        print('proj_probs44: ', proj_probs[i].shape)
                        write_ply(test_name2,
                                  [points, proj_probs[i]],
                                  ['x', 'y', 'z'] + prob_names)

                        # Save potentials
                        pot_points = np.array(test_loader.dataset.pot_trees[i].data, copy=False)
                        pot_name = join(test_path, self.ck_name, 'potentials', cloud_name)
                        pots = test_loader.dataset.potentials[i].numpy().astype(np.float32)
                        write_ply(pot_name,
                                  [pot_points.astype(np.float32), pots],
                                  ['x', 'y', 'z', 'pots'])

                        # Save ascii preds
                        if test_loader.dataset.set == 'test':
                            ascii_name = join(test_path, self.ck_name, 'predictions', cloud_name[:-4] + '.txt')
                            np.savetxt(ascii_name, preds, fmt='%d')

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))


            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break

        return


