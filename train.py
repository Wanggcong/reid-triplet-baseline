# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from model import ft_net, ft_net_dense, resnet50
from random_erasing import RandomErasing
import json
import shutil
import shuttle
from sampler import MySampler
from triplet import TripletLoss
from optimizer import train_model

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir',default='/home/wang/datasets/reid/Market-1501-v15.09.15/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--num_epochs', default=60, type=int, help='num_epochs')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
opt = parser.parse_args()
data_dir = opt.data_dir
name = opt.name
num_epochs = opt.num_epochs
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
#print(gpu_ids[0])


######################################################################
# Load Data
transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize(144, interpolation=3),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list + [RandomErasing(opt.erasing_p)]
    
if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)



data_transforms = transforms.Compose( transform_train_list)


# train_all = ''
# image_datasets = datasets.ImageFolder(os.path.join(data_dir,train_all),data_transforms)
# dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,shuffle=True, num_workers=4)
# use_gpu = torch.cuda.is_available()

train_all = 'train_all'
image_datasets = datasets.ImageFolder(os.path.join(data_dir,train_all),data_transforms)

idx_dict = {}
for i in range(len(image_datasets)):
    idx = image_datasets.imgs[i][1]
    idx_dict.setdefault(idx,[])
    idx_dict[idx].append(i)
sampler = MySampler(idx_dict,opt.batchsize,len(image_datasets))
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize, sampler=sampler, num_workers=4)

# image_datasets = Preprocessor(root=data_dir, transform=data_transforms)
# sampler = MySampler(my_dict_frames, my_dict_indexes,centers,frame_knn,points_current, k_points_other_frame,frames_per_epoch)
# dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize, sampler=sampler, num_workers=4)
# #  pin_memory=True, drop_last=True)


# dataset_sizes = len(pseudo_labels)
use_gpu = torch.cuda.is_available()
# inputs, classes = next(iter(dataloaders))
inputs = next(iter(dataloaders))


######################################################################
class_names = 128
# # step3: Finetuning the convnet
# Load a pretrainied model and reset final fully connected layer.
if opt.use_dense:
    # model = ft_net_dense(len(class_names))
    model = ft_net_dense(class_names)
    print('********************ft_net_dense******************')
else:
    # model = ft_net(len(class_names))
    # model = resnet50(len(class_names))    
    model = resnet50(class_names)    
    # model = resnet50(751)    
    print('********************ft_net************************')
print(model)

if use_gpu:
    model = model.cuda()

# if opt.load_flag:
#     save_path = os.path.join('./model',opt.load_name,'net_%s.pth'%opt.which_epoch)
#     model.load_state_dict(torch.load(save_path))

######################################################################
# step 4: setting and train
# criterion = nn.CrossEntropyLoss()
criterion = TripletLoss(margin=0.5).cuda()
ignored_params = list(map(id, model.fc1.parameters() )) + list(map(id, model.fc2.parameters() ))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.01},
             {'params': model.fc1.parameters(), 'lr': 0.1},
             {'params': model.fc2.parameters(), 'lr': 0.1}
         ], momentum=0.9, weight_decay=5e-4, nesterov=True)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
dir_name = os.path.join('./model',name)

if not os.path.isdir('./model'):
    os.mkdir('./model')
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
# save opts
with open('%s/opts.json'%dir_name,'w') as fp:
    json.dump(vars(opt), fp, indent=1)
model = train_model(model,dataloaders, use_gpu, criterion, optimizer_ft, exp_lr_scheduler,num_epochs, name, gpu_ids[0])

