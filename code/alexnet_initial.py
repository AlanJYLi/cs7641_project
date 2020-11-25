import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models



class AlexNet_Initial(nn.Module):
    def __init__(self,n_classes=10):
        
        super(AlexNet_Initial, self).__init__()
        
        self.alex_initial =models.alexnet(pretrained=False)
        num_in = self.alex_initial.classifier[6].in_features
        self.alex_initial.classifier[6] = nn.Linear(num_in,n_classes)
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')


    def forward(self, images):
        
        scores = None
        input_img = images = F.interpolate(images, size=(96, 128))
        scores = self.alex_initial(input_img)
        return scores
