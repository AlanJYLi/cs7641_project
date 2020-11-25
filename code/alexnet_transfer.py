import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models



class AlexNet_Transfer(nn.Module):
    def __init__(self,n_classes=10):
        '''
        AlexNet
        '''
        super(AlexNet_Transfer, self).__init__()
        pretrainedalexnet =models.alexnet(pretrained=True)
        for para in pretrainedalexnet.parameters():
            para.requires_grad = False
        self.cnn_layers = pretrainedalexnet.features
        self.avg_pool = nn.AdaptiveAvgPool2d(6)
        self.flat=nn.Flatten()
        self.fc1 = nn.Linear(int(256*6*6), 4096)
        self.relu1 = nn.ReLU()
        self.fcdrop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU()
        self.fcdrop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, n_classes)
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')


    def forward(self, images):
        
        scores = None
       
        scores = self.cnn_layers(images)
        scores = self.avg_pool(scores)
        scores = self.flat(scores)
        scores = self.fc1(scores)
        scores = self.relu1(scores)
        scores = self.fcdrop1(scores)
        scores = self.fc2(scores)
        scores = self.relu2(scores)
        scores = self.fcdrop2(scores)
        scores = self.fc3(scores)
        return scores
