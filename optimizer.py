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
import json
import shutil
import shuttle

def train_model(model, dataloaders, use_gpu, criterion, optimizer, scheduler, num_epochs, name, gpu_id):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        model.train(True)  # Set model to training mode
        running_loss = 0.0
        # running_corrects = 0
        data_size = 0.0

        # Iterate over data.
        for data in dataloaders:
            # get the inputs
            inputs, labels = data
            # inputs = data
            # labels = targets
            # labels = torch.FloatTensor(labels)
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            # print('labels:',labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            # _, preds = torch.max(outputs.data, 1)
            loss, prec = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.data.item()
            # running_corrects += torch.sum(preds == labels.data)
            # print('data :',data)
            data_size += data[1].size()[0]


        epoch_loss = running_loss*1.0 / data_size
        # epoch_acc = running_corrects / dataset_sizes
        print('{} Loss: {:.4f}'.format('train', epoch_loss))
        if epoch%10 == 9:
            save_network(model, epoch, name, gpu_id)
        # y_loss.append(epoch_loss)
        # y_err.append(1.0-epoch_acc)            

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    # model.load_state_dict(last_model_wts)
    save_network(model, 'last',name,gpu_id)
    return model



# def draw_curve(current_epoch):
#     x_epoch.append(current_epoch)
#     ax0.plot(x_epoch, y_loss, 'bo-', label='train')
#     ax1.plot(x_epoch, y_err, 'bo-', label='train')
#     if current_epoch == 0:
#         ax0.legend()
#         ax1.legend()
#     fig.savefig( os.path.join('./model',name,'train.jpg'))


######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label,name,gpu_id):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_id)
