from __future__ import absolute_import
from __future__ import print_function
from collections import defaultdict
import numpy as np
import torch
import os.path as osp
import random
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)
import os
import pickle

class MySampler(Sampler):
    def __init__(self, idx_dict, imgs_per_batch, imgs):
        self.idx_dict = idx_dict
        self.imgs_per_batch = imgs_per_batch
        self.imgs = imgs
        self.ids = len(idx_dict)
        self.imgs_per_idx = 4
        self.ids_per_batch = imgs_per_batch//self.imgs_per_idx
        self.batches_per_epoch = imgs//imgs_per_batch  
        self.indices = torch.randperm(self.ids)         
        self.iter = 0 
    
    def __len__(self):
        return self.batches_per_epoch*self.imgs_per_batch
    def __iter__(self):
        ret = []
        for _ in range(self.batches_per_epoch):
            for _ in range(self.ids_per_batch):
                if(self.iter>=self.ids):
                    self.iter = 0
                    self.indices = torch.randperm(self.ids)
                idx = self.indices[self.iter]
                # print('idx:',idx.item())
                idx_imgs = self.idx_dict[idx.item()]
                self.iter+=1                   
                if len(idx_imgs) >= self.imgs_per_idx:
                    idx_imgs_sample = np.random.choice(idx_imgs, size=self.imgs_per_idx, replace=False)
                else:
                    idx_imgs_sample = np.random.choice(idx_imgs, size=self.imgs_per_idx, replace=True)
                ret.extend(idx_imgs_sample)
        return iter(ret)




# class MySampler(Sampler):
#     def __init__(self, my_dict_frames, my_dict_indexes, centers, frame_knn=5, points_current=1, k_points_other_frame =1, frames_per_epoch=100):
#         self.centers = centers
#         self.my_dict_frames = my_dict_frames
#         self.my_dict_indexes = my_dict_indexes
#         self.frame_knn = frame_knn
#         self.points_current = points_current
#         self.k_points_other_frame = k_points_other_frame
#         # self.frames_num = len(my_dict_indexes) #
#         self.frames_num = 100 #
#         self.frames_per_epoch = self.frames_num
#         ##### because the key of my_dict_indexes begins with 1, not 0; thus we plus 1  
#         self.indices = torch.randperm(self.frames_num-2*self.frame_knn)+self.frame_knn+1            
#         self.iter = 0 #
#     def __len__(self):
#         return (2*self.frame_knn*self.k_points_other_frame*self.points_current+self.points_current)*self.frames_per_epoch
#     def __iter__(self):
#         ret = []
#         for _ in range(self.frames_per_epoch):
#             if(self.iter>=self.frames_num-2*self.frame_knn):
#                 self.iter = 0
#                 self.indices = torch.randperm(self.frames_num-2*self.frame_knn)+self.frame_knn+1
#             frame = self.indices[self.iter]
#             self.iter+=1
#             # computer knn with two steps:              
            
#             # step 1: k farthest points (kfp) in center frame: try random
#             points_indexes = self.my_dict_indexes[frame]         
#             if len(points_indexes) >= self.points_current:
#                 points_selected_indexes = np.random.choice(points_indexes, size=self.points_current, replace=False)
#             else:
#                 points_selected_indexes = np.random.choice(points_indexes, size=self.points_current, replace=True)
#             # step 2: knn points in knn frames
#             indexes=[]
#              ### not one point, need one more for below
#             for point_index in points_selected_indexes:
#                 # point = self.my_dict_frames[frame][point_index]
#                 point = self.centers[point_index]
#                 for i in range(frame-self.frame_knn, frame+self.frame_knn+1):
#                     if i != frame:
#                         points_cmp = self.my_dict_frames[i]
#                         # return points 
#                         sorted_ind = np.argsort(np.sum(pow(np.array(points_cmp) - np.array(point),2),axis=1),axis=0)
#                         indexes = indexes + [self.my_dict_indexes[i][ind] for ii, ind in enumerate(sorted_ind) if ii<self.k_points_other_frame] 
            
#             # batch_indexes = points_indexes+indexes
#             batch_indexes = points_selected_indexes.tolist()+indexes
#             ret.extend(batch_indexes)
#         print('ret len:',len(ret))
#         return iter(ret)

