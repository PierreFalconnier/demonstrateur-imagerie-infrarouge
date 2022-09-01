from tkinter import W
import torch
from torchvision.transforms import ToTensor, Resize, RandomHorizontalFlip, RandomRotation,RandomPerspective, RandomAffine
from torch.nn import Threshold
from typing import Any, Dict, List, Tuple, Type, Callable, Optional
import random

class MyOneHotEncoding(object) :
    """
    Encode the target label as a binary vector of size (1,10)

    Attributs:
        identity (torch.tensor): Identity matrix used for encoding label
    """
    def __init__(self)-> None:
        self.identity = torch.eye(4)

    def __call__(self,sample)->dict:
        sample['y']= self.identity[sample['y'],:]
        return sample

class MyPilToTensor(object):
    """
    Convert a PIL image as a torch.Tensor

    Attributs:
        TrToTensor (class): Convert a PIL image as a torch.Tensor
    """
    def __init__(self)-> None:
        self.TrToTensor=ToTensor()

    def __call__(self,sample)-> dict:
        sample['x']= self.TrToTensor(sample['x'])
        return sample

class MyFloat16(object):
    """
    Cast a torch.Tensor as torch.float16
    """
    def __call__(self,sample)-> dict:
        sample['x']= sample['x'].to(torch.float16)
        return sample

class MyResize(object):
    """
    Resize a tensor
    """
    def __init__(self) -> None:
        self.TrResize = Resize(size = (64,64))


    def __call__(self, sample) -> dict:
        sample["x"] = self.TrResize(sample["x"])
        return sample
        

class MyThreshold(object):

    def __init__(self) -> None:
        self.threshold = 0.2
    
    def __call__(self, sample) -> dict :
        sample["x"] = torch.where(sample["x"]> self.threshold, 1.0, 0.0 )
        return sample

# Data augmentation


class MyRandomHorizontalFlip(object):

    def __init__(self) -> None:
        self.proba = 0.5
        self.TrRandomVerticalFlip = RandomHorizontalFlip(p=self.proba)
    
    def __call__(self, sample) -> dict :
        sample["x"] =self.TrRandomVerticalFlip(sample["x"])
        return sample


class MyRandomRotation():
    def __init__(self) -> None:
        self.proba = 0.5
        self.TrRandomRotation = RandomRotation(degrees=15)
    
    def __call__(self, sample) -> dict :
        if random.uniform(0, 1)<self.proba :
            sample["x"] =self.TrRandomRotation(sample["x"])
        return sample


class MyRandomPerspective():
    def __init__(self) -> None:
        self.proba = 0.1
        self.TrRandomPerspective = RandomPerspective(distortion_scale=0.05, p=self.proba)
    
    def __call__(self, sample) -> dict :
        sample["x"] =self.TrRandomPerspective(sample["x"])
        return sample

class MyRandomAffine():
    def __init__(self) -> None:
        self.proba = 0.7
        self.TrRandomAffine = RandomAffine(degrees=10, translate=(0.3, 0.3), scale=(0.8,1.2))  # possible d'ajouter scale, possède pas d'arugment rotation
    
    def __call__(self, sample) -> dict :
        if random.uniform(0, 1)<self.proba :
            sample["x"] =self.TrRandomAffine(sample["x"])
        return sample