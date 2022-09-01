from tkinter import W
import torch
from torchvision.transforms import ToTensor, Resize
from typing import Any, Dict, List, Tuple, Type, Callable, Optional

class OneHotEncoding(object) :
    """
    Encode the target label as a binary vector of size (1,10)

    Attributs:
        identity (torch.tensor): Identity matrix used for encoding label
    """
    def __init__(self)-> None:
        self.identity = torch.eye(10)

    def __call__(self,sample)->dict:
        sample['y']= self.identity[sample['y'],:]
        return sample

class PilToTensor(object):
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

class Float16(object):
    """
    Cast a torch.Tensor as torch.float16
    """
    def __call__(self,sample)-> dict:
        sample['x']= sample['x'].to(torch.float16)
        return sample

class ResizeImage(object):
    """
    Resize a tensor
    """
    def __init__(self) -> None:
        self.TrResize = Resize(size = (128,128))


    def __call__(self, sample) -> dict:
        sample["x"] = self.TrResize(sample["x"])
        return sample
        

# class Image2Vector(object):
#     """
#     View (Reshape) an image as a vector.

#     Attributs:
#         nbPixel (int): total amount pixel. 
#     """
#     def __init__(self):
#         # mnist images are (1, 28, 28) (channels, width, height)
#         self.nbPixel = 1*28*28
    
#     def __call__(self,sample)-> dict:
#         sample['x']= sample['x'].view(self.nbPixel)
#         return sample


