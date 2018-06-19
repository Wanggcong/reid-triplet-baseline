from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
# import torch.autograd.functional as F

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

class ft_net(nn.Module):

    def __init__(self, class_num ):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        num_ftrs = model_ft.fc.in_features
        add_block = []
        num_bottleneck = 512
        add_block += [nn.Linear(num_ftrs, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]  #default dropout rate 0.5
        #transforms.CenterCrop(224),
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        model_ft.fc = add_block
        self.model = model_ft

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x


class ft_net_dense(nn.Module):

    def __init__(self, class_num ):
        super(ft_net_dense, self).__init__()
        model_ft = models.densenet121(pretrained=True)
        # add pooling to the model
        # in the originial version, pooling is written in the forward function 
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))

        add_block = []
        num_bottleneck = 512
        add_block += [nn.Linear(1024, num_bottleneck)]  #For ResNet, it is 2048
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        model_ft.fc = add_block
        self.model = model_ft

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.model.features(x)  
        x = x.view(x.size(0),-1)
        x = self.model.fc(x)
        x = self.classifier(x)
        return x

###################################################################################################################
# from __future__ import absolute_import
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

# from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, pretrained=True, cut_at_pooling=False, \
        num_features=0, norm=False, dropout=0, num_classes=128, block=Bottleneck, layers=[3, 4, 6, 3]):
        # BasicBlock
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

                    
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        print('block.expansion:',block.expansion)
        self.embedding = 512
        self.fc1 = nn.Linear(512 * block.expansion, self.embedding)
        self.bn2 =nn.BatchNorm1d(self.embedding)
        # self.relu2 =nn.LeakyReLU()
        self.relu2 =nn.ReLU()
        # self.dropout2 =nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(self.embedding, num_classes)


        # init.kaiming_normal(self.fc1.weight.data, a=0, mode='fan_out')   #fc1
        init.kaiming_normal(self.fc1.weight, mode='fan_out')
        init.constant(self.fc1.bias, 0)

        # init.normal(self.bn2.weight.data, 1.0, 0.02)                   #bn2
        init.constant(self.bn2.weight, 1)
        init.constant(self.bn2.bias, 0)

        init.normal(self.fc2.weight, std=0.001)                   #fc2
        init.constant(self.fc2.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print('pre size:',x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print('pre size:',x.size())
        # x_rnn = x


        x = self.layer1(x)
        #print('layer1 size:',x.size())

        x = self.layer2(x)
        #print('layer2 size:',x.size())

        x = self.layer3(x)
        #print('layer3 size:',x.size())

        x = self.layer4(x)
        #print('layer4 size:',x.size())


        #########################################
        # rnn, used as a mask
        # print('lwq:',x_rnn.size())
        # print('lwqlwq:',x.size())

        
        # print('lwqlwqlwq:',x.size())

        # print('wgc x new size:', x.size())
        
        # x,_ = torch.max(x,1,keepdim=True)
        # x = F.normalize(x)
        # x = F.tanh(x)
        # print('wgcwgchhhhh:',x)
        
        # x = F.upsample(x, x_rnn.size()[2:4], mode='bilinear')
        # x = torch.mean(x,1,keepdim=True)
        # x = torch.add(x_rnn,x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        #########################################

        # x = self.avgpool(x)
        x = self.avgpool(x)
        # x = F.avg_pool2d(x, x.size()[2:])

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        return x


def resnet18(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        updated_params = model_zoo.load_url(model_urls['resnet34'])
        new_params = model.state_dict()
        new_params.update(updated_params)
        model.load_state_dict(new_params)    
    return model


def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    if pretrained:
        updated_params = model_zoo.load_url(model_urls['resnet50'])
        updated_params.pop('fc.weight')
        updated_params.pop('fc.bias')
        # print('updated_params:',updated_params)
        # print('wgcwgcwgc:',type(updated_params))
        new_params = model.state_dict()
        new_params.update(updated_params)
        model.load_state_dict(new_params)
    # else:
    if False:
        print('***************************wgc will succeed!********************************8')
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

        init.kaiming_normal(model.fc1.weight, mode='fan_out')
        init.constant(model.fc1.bias, 0)
        init.constant(model.bn2.weight, 1)
        init.constant(model.bn2.bias, 0)

        init.normal(model.fc2.weight, std=0.001)
        init.constant(model.fc2.bias, 0)
    return model


def resnet101(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


#####################################################################################################################
# debug model structure
net = ft_net(751)
# net = ft_net_dense(751)
#print(net)
input = Variable(torch.FloatTensor(8, 3, 224, 224))
output = net(input)
print('net output size:')
# print(output.shape)
print(output.size())
