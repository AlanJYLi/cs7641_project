import torch
import torch.nn as nn
import numpy as np
from torchvision.models import vgg16_bn

class Vgg16_pretrain(nn.Module):
  '''train based on pretrained weight'''

  def __init__(self, n_classes=10):
    '''
    Init function to define the layers and loss function
    '''
    super().__init__() 
    model = vgg16_bn(pretrained=True)
    for para in model.parameters():
        para.requires_grad = False
    self.cnn_layers = model.features

    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(int(512*3*4), 4096)
    self.relu1 = nn.ReLU()
    self.fcdrop1 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(4096, 4096)
    self.relu2 = nn.ReLU()
    self.fcdrop2 = nn.Dropout(0.5)
    self.fc3 = nn.Linear(4096, n_classes)

    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')
    

  def forward(self, x):
    '''
    Perform the forward pass with the net
    '''
    res = self.cnn_layers(x)
    res = self.flat(res)
    res = self.fc1(res)
    res = self.relu1(res)
    res = self.fcdrop1(res)
    res = self.fc2(res)
    res = self.relu2(res)
    res = self.fcdrop2(res)
    res = self.fc3(res)
    return res
