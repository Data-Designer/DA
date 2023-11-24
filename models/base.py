#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/14 19:20
# @Author  : Jack Zhao
# @Site    : 
# @File    : base.py
# @Software: PyCharm

# #Desc: 

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

import torch
import torch.nn as nn

from config import opt
from torchvision import models
from torch.autograd import Function
from torchvision import models



class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output * -self.lambd 


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


class GeneratorRes(nn.Module):
    def __init__(self,option='resnet18', pret=False):
        super(GeneratorRes, self).__init__()
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)

        mod = list(model_ft.children())
        mod.pop() 
        self.features = nn.Sequential(*mod)
#         self.bottleneck = nn.Linear(model_ft.fc.in_features, 256)

#         
#         nn.init.normal_(self.bottleneck.weight.data, 0, 0.005)
#         nn.init.constant_(self.bottleneck.bias.data, 0.1)
        self.__in_features = model_ft.fc.in_features

#         self.dim = 256

    def forward(self, x):
        out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.bottleneck(out)
#         out = out.view(out.size(0), self.dim) 

        return out

    def output_dim(self):
        return self.__in_features



class LeNet(nn.Sequential):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__(
            nn.Conv2d(3, 20, kernel_size=5), 
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.num_classes = num_classes
        self.out_features = 500

    def copy_head(self):
        return nn.Linear(500, self.num_classes)

    def output_dim(self):
        return self.out_features

def lenet(pretrained=False, **kwargs):
    return LeNet(**kwargs)



class DTN(nn.Sequential):
    def __init__(self, num_classes=10):
        super(DTN, self).__init__(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.num_classes = num_classes
        self.out_features = 512

    def copy_head(self):
        return nn.Linear(512, self.num_classes)

    def output_dim(self):
        return self.out_features



def dtn(pretrained=False, **kwargs):
    return DTN(**kwargs)



# class DTN(nn.Module):
#     def __init__(self):
#         super(DTN, self).__init__()
#         self.conv_params = nn.Sequential (
#                 nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
#                 nn.BatchNorm2d(64),
#                 nn.Dropout2d(0.1),
#                 nn.ReLU(),
#                 nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
#                 nn.BatchNorm2d(128),
#                 nn.Dropout2d(0.3),
#                 nn.ReLU(),
#                 nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
#                 nn.BatchNorm2d(256),
#                 nn.Dropout2d(0.5),
#                 nn.ReLU()
#                 )
    
#         self.fc_params = nn.Sequential (
#                 nn.Linear(256*4*4, 512),
#                 nn.BatchNorm1d(512),
#                 nn.ReLU(),
#                 nn.Dropout()
#                 )

#         self.classifier = nn.Linear(512, 10)
#         self.__in_features = 512

#     def forward(self, x):
#         x = self.conv_params(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_params(x)
# #         y = self.classifier(x)
#         return x

#     def output_dim(self):
#         return self.__in_features


    

class Discriminator(nn.Module):
    def __init__(self, num_cls=12, num_layer=2, num_unit=2048, prob=0.5, middle=1000):
        super(Discriminator, self).__init__()
        layers = []

        # MLP
        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(num_unit, middle))
        layers.append(nn.BatchNorm1d(middle, affine=True))
        layers.append(nn.ReLU(inplace=True))

        for i in range(num_layer - 1):
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(middle, middle))
            layers.append(nn.BatchNorm1d(middle, affine=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(middle, num_cls)) # classifier logit
        self.classifier = nn.Sequential(*layers)

    def set_lambda(self, lambd):
        self.lambd = lambd


    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x


class BaseNetwork(nn.Module):
    def __init__(self, option, pretrain, use_bottleneck, device):
        super(BaseNetwork,self).__init__()
        self.option = option
        self.use_bottleneck = use_bottleneck
        # model definition
        if option == "dtn":
            self.G = dtn() # pretrained=pretrain
        elif option == "lenet":
            self.G = lenet() # pretrained=pretrain
        else:
            option = 'resnet' + str(opt.RESNET)
            self.G = GeneratorRes(option,pret=pretrain)
        
        g_dim = self.G.output_dim()
        
        self.g_dim = g_dim

#         self.bottleneck = nn.Identity().to(device)

        if use_bottleneck: 
#             self.bottleneck = nn.Linear(self.g_dim, opt.BOTTLE).to(device)
            self.bottleneck_1 = nn.Linear(g_dim, opt.BOTTLE).to(device)
            self.bottleneck_2 = nn.Linear(g_dim, opt.BOTTLE).to(device)
            dim = opt.BOTTLE
        else:
#             self.pool_layer_1 = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(output_size=(1, 1)),
#                 nn.Flatten()
#             )
#             self.pool_layer_2 = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(output_size=(1, 1)),
#                 nn.Flatten()
#             )
            dim = g_dim
        

            
        self.F1 = Discriminator(num_cls=opt.CLASSNUM, num_layer=opt.DISLAYER, num_unit=dim, middle=1024)
        self.F2 = Discriminator(num_cls=opt.CLASSNUM, num_layer=opt.DISLAYER, num_unit=dim, middle=1024)
        self.F1.apply(self.init_weights)
        self.F2.apply(self.init_weights)
        self.G, self.F1, self.F2 = self.G.to(device), self.F1.to(device), self.F2.to(device)
        
#     def weights_init(m):
#         """初始化"""
#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1:
#             m.weight.data.normal_(0.0, 0.01)
#             m.bias.data.normal_(0.0, 0.01)
#         elif classname.find('BatchNorm') != -1:
#             m.weight.data.normal_(1.0, 0.01)
#             m.bias.data.fill_(0)
#         elif classname.find('Linear') != -1:
#             m.weight.data.normal_(0.0, 0.01)
#             m.bias.data.normal_(0.0, 0.01)
            
    def init_weights(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)
        elif classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
            
    def get_generator_learner_parameters(self):
        parameter_list = [{"params": self.G.parameters()}
                         ] # 有出入
        return parameter_list
    

    def get_classifier_parameters(self):
        if self.use_bottleneck:
            parameter_list = [{"params": self.F1.parameters()},
                             {"params": self.F2.parameters()},{"params": self.bottleneck_1.parameters()},{"params": self.bottleneck_2.parameters()}]
        else:
#             parameter_list = parameter_list = [{"params": self.F1.parameters()},
#                              {"params": self.F2.parameters()},{"params": self.pool_layer_1.parameters()},{"params": self.pool_layer_2.parameters()}]
            parameter_list = parameter_list = [{"params": self.F1.parameters()},
                             {"params": self.F2.parameters()}]
        return parameter_list

            
    def forward(self, x):
        x = self.G(x) # torch.Size([144, 3, 64, 64])
        x = x.view(x.size(0), -1)

        if self.use_bottleneck:
            x_1 = self.bottleneck_1(x)
            x_2 = self.bottleneck_2(x)
        else:
#             x_1 = self.pool_layer_1(x)
#             x_2 = self.pool_layer_2(x)
            x_1 = x
            x_2 = x
            
        y1 = self.F1(x_1)
        y2 = self.F2(x_2)
        return x, y1, y2
        
        
        
    
        


if __name__ == '__main__':
    # model = GeneratorRes()
    # print(model.output_dim())
#     model = lenet(pretrained=False)
#     print(model(torch.randn(32,3,28,28)).shape)

    from config import opt
    model = BaseNetwork(option='resnet', pretrain=False, use_bottleneck=False, device = 'cuda:0')
    print(model(torch.randn(32,3,28,28).cuda())[1].shape)


