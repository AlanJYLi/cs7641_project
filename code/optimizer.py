import torch

def get_optimizer(model, parameters):
  '''
  Returns the optimizer.
  '''
  optimizer_type = parameters["optimizer_type"]
  learning_rate = parameters["lr"]
  weight_decay = parameters["weight_decay"]

  if optimizer_type == 'sgd':
    momentum = parameters["momentum"]
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

  if optimizer_type == 'adam':
    betas = parameters["betas"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=1e-08, weight_decay=weight_decay)

  return optimizer
