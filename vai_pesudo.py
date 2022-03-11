#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare pseudo labels to train the segmentation network
"""

from utils.mayavi_visu import *
import pickle
from sklearn.neighbors import NearestNeighbors
import  glob

base_folder = '.../test'
model_n = 
t_list = [0, 1, 2]
ck = # 'chkp_**'
for t in t_list:
    
    base_path = join(base_folder, model_n, ck)
    fn_list = glob.glob(base_path+'/predictions/*.ply')
    # compare normal .ply and lbs.ply
    
    for fn in fn_list:
        data = read_ply(fn)
        pts_lbs = np.array([data['x'],data['y'],data['z']]).T
        pseudo_lbs = data['preds']
        
        ff = fn.split('/')[-1].split('.pts.ply')[0]
        fnfn = #'/.ply'
        data_sub = read_ply(fnfn)
        pts_sub = np.array([data_sub['x'],data_sub['y'],data_sub['z']]).T
        lbs = data_sub['class']
        
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pts_lbs[:,:3])
        distance, indices = nbrs.kneighbors(pts_sub[:,:3])    
        indices = np.squeeze(indices)
        
        prob_p=join(base_path, 'probs/Vaihingen3D_Train.pts.ply')
        data = read_ply(prob_p)       
        probs = np.vstack((data['Powerline'], data['Low_vegetation'], data['Impervious_surfaces'],
                            data['Car'], data['Fence/Hedge'], data['Roof'], data['Facade'],
                            data['Shrub'], data['Tree'])).T        
                
        region_class = np.genfromtxt('class_lb.txt', delimiter=' ')
        probs = probs[indices]
        probs = probs*region_class
        
        empty = np.max(probs, axis=-1)<(0.1*t)
        pseudo_lbs = pseudo_lbs[indices]
        pseudo_lbs[empty]=10
        uu, cc = np.unique(pseudo_lbs, return_counts=True) 

        cc = cc[:9]
        w = np.log(1/(cc/np.sum(cc)))
        w_n = w/np.sum(w)        
        
        new_lbs = pseudo_lbs#[indices]
        
        save_path = join(base_folder, model_n, ff+'_'+ck+'_'+str(t)+'reglb'+'_pseudo.txt')
        np.savetxt(save_path, new_lbs, fmt='%i')
        w_path = join(base_folder, model_n, ff+'_'+ck+'_'+str(t)+'reglb'+'_weight.txt')
        np.savetxt(w_path, w_n, fmt='%.3f')
        print(fn)     
  
        

        
        
        
        