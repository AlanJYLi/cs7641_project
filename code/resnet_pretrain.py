import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet34

class Resnet_pretrain(nn.Module):
  '''train based on pretrained weight'''

  def __init__(self, n_classes=10):
    '''
    Init function to define the layers and loss function
    '''
    super().__init__()
    self.model = resnet34(pretrained=True)
    for para in self.model.parameters():
        para.requires_grad = False
    
    self.model.fc = nn.Linear(self.model.fc.in_features, 1024)
    self.relu = nn.ReLU()
    self.fcdrop = nn.Dropout(0.5)
    self.fc2 = nn.Linear(1024, 2048)
    self.relu2 = nn.ReLU()
    self.fcdrop2 = nn.Dropout(0.5)
    self.fc3 = nn.Linear(2048, 4096)
    self.relu3 = nn.ReLU()
    self.fcdrop3 = nn.Dropout(0.5)
    self.fc4 = nn.Linear(4096, n_classes)

    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')
    

  def forward(self, x):
    '''
    Perform the forward pass with the net
    '''
    res = self.model(x)
    res = self.relu(res)
    res = self.fcdrop(res)
    res = self.fc2(res)
    res = self.relu2(res)
    res = self.fcdrop2(res)
    res = self.fc3(res)
    res = self.relu3(res)
    res = self.fcdrop3(res)
    res = self.fc4(res)
    return res
