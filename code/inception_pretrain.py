import torch
import torch.nn as nn
import numpy as np
from torchvision.models import inception_v3

class Inception_pretrain(nn.Module):
  '''train based on pretrained weight'''

  def __init__(self, n_classes=10):
    '''
    Init function to define the layers and loss function
    '''
    super().__init__()
    self.model = inception_v3(pretrained=True, aux_logits=False)
    for para in self.model.parameters():
        para.requires_grad = False
    
    self.model.fc = nn.Linear(self.model.fc.in_features, 4096)
    self.relu = nn.ReLU()
    self.fcdrop = nn.Dropout(0.5)
    self.fc2 = nn.Linear(4096, 4096)
    self.relu2 = nn.ReLU()
    self.fcdrop2 = nn.Dropout(0.5)
    self.fc3 = nn.Linear(4096, n_classes)

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
    return res
