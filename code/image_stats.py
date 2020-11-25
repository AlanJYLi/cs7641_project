import glob
import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def get_mean_and_std(dir_name):
  '''
  Compute the mean and the standard deviation of the dataset.
  Scale to [0,1] before computing mean and standard deviation
  '''
  scale_r = StandardScaler()
  scale_g = StandardScaler()
  scale_b = StandardScaler()
  for root, dirs_name, files_name in os.walk(dir_name):
    for jpg in files_name:
      jpg_address = os.path.join(root, jpg)
      img = Image.open(jpg_address)
      img = np.asarray(img)
      img = img / 255.0
      img_r = img[:, :, 0].reshape((-1,1))
      img_g = img[:, :, 1].reshape((-1,1))
      img_b = img[:, :, 2].reshape((-1,1))
      scale_r.partial_fit(img_r)
      scale_g.partial_fit(img_g)
      scale_b.partial_fit(img_b)
  mean = [scale_r.mean_[0], scale_g.mean_[0], scale_b.mean_[0]] 
  std = [np.sqrt(scale_r.var_)[0], np.sqrt(scale_g.var_)[0], np.sqrt(scale_b.var_)[0]]
  return mean, std
