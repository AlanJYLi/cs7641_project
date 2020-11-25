import glob
import os
import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import torchvision.transforms as transforms


class ImageLoader(data.Dataset):
  '''
  Class for data loading
  '''
  train_folder = 'train'
  test_folder = 'test'

  def __init__(self, root_dir, split='train', resize=(96, 128)):
    '''
    Init function for the class.
    '''
    self.root = os.path.expanduser(root_dir)
    self.transform = transforms.Compose([transforms.Resize(resize),
    	                                 transforms.ToTensor(),
    	                                 transforms.Normalize([0.31437435026332544, 0.3805657139133703, 0.3733153854924628], 
                                                              [0.29172749892553, 0.3331829868413413, 0.3351285321390462])
                                         ])
    self.split = split

    if split == 'train':
      self.curr_folder = os.path.join(root_dir, self.train_folder)
    elif split == 'test':
      self.curr_folder = os.path.join(root_dir, self.test_folder)

    self.class_dict = self.get_classes()
    self.dataset = self.load_imagepaths_with_labels(self.class_dict)


  def load_imagepaths_with_labels(self, class_labels):
    '''
    Fetches all image paths along with labels
    
    class_labels: the class labels dictionary, with keys being the classes and the values being the class index.
    '''
    img_paths = []
    for name in class_labels:
        class_folder = os.path.join(self.curr_folder, name)
        jpgs = os.listdir(class_folder)
        for jpg in jpgs:
            img_paths.append((os.path.join(class_folder, jpg), class_labels[name]))
    return img_paths


  def get_classes(self):
    '''
    Get the classes (which are folder names in self.curr_folder) along with their associated integer index.
    '''
    classes = dict()
    names = os.listdir(self.curr_folder)
    names.sort()
    for i, name in enumerate(names):
        classes[name] = i
    return classes


  def load_img_from_path(self, path):
    ''' 
    Loads the image 
    '''
    img = Image.open(path)
    return img


  def __getitem__(self, index):
    '''
    Fetches the item (image, label) at a given index
    '''
    filepath, class_idx = self.dataset[index]
    img = self.load_img_from_path(filepath)
    img = self.transform(img)
    return img, class_idx


  def __len__(self) -> int:
    """
    Returns the number of items in the dataset
    """
    return len(self.dataset)
