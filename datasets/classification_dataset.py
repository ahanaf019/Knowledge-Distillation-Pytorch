import torch
from torch.utils.data import Dataset
from typing import Callable
from utils.utils import read_image


class ClassificationDataset(Dataset):
    def __init__(self,images, labels, image_size,  db_path_root=None, augments: Callable=None, transforms: Callable=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.augments = augments
        self.transforms = transforms
        self.image_size = image_size
        self.db_path_root = db_path_root
    
    
    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index):
        image = read_image(self.images[index], self.image_size)
        label = self.labels[index]

        if self.augments is not None:
            image = self.augments(image=image)['image']
        
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label