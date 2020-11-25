import torch
import torch.nn as nn
import numpy as np
from torchvision.models import inception_v3

class Inception_initial(nn.Module):
  '''train on our dataset'''

  def __init__(self, n_classes=10):
    '''
    Init function to define the layers and loss function
    '''
    super().__init__()
    self.model = inception_v3(pretrained=False, aux_logits=False)
    self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')
    

  def forward(self, x):
    '''
    Perform the forward pass with the net
    '''
    res = self.model(x)
    return res
