import os, sys
from typing import Any, Dict, List, Tuple, Type, Callable, Optional, Union
import numpy as np
from torch.utils.data.dataset import Dataset
from glob import glob
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np



#Path
CUR_DIR_PATH=os.path.dirname(__file__)

class DataSetHgr(Dataset) :
    """
    A Dataset HANDLES the data, a DataLoader ORGANIZES the data. A dataset does not need to know if the data comes from the train, test or validation.
    Its role is to extract 1 sample from a dataset and apply a list of transforms. In other words, it creates the sample needed for a model.

    Here we create a Map-style dataset. Never put your sample on the GPU with .cuda() ! It is more efficient to put on GPU a batch of samples directly,
    but with pytorch lightning (pl) you should never manage the device location of each tensor because pl manage this things for you.

    Attributs:
        data (list): Contains the paths of all the images from the Raw folder
        transform (list): Contains all the transforms will be applied on each samples
    """
    def __init__(self,transform: Optional[Callable] = None)-> None:
        """
            Constructor

            Args:
                transform (list): Contains all the transforms will be applied on each samples
        """
        super(Dataset,self).__init__()
        self.data = glob(os.path.join(self.raw_folder,'*/*.png'))
        self.transform=transform
        print(os.path.basename(self.data[8]))

    def __getitem__(self, index: int)->dict:
        """
        This method return a sample.

        Args:
            index (int): Index of the number to be extracted from the dataset.
        
        Returns:
            sample (Dict): Each sample is returned as a Dict with keys 'x': image, 'y': label, 'fileName': name of image.
        """

        dirName = os.path.basename(os.path.dirname(self.data[index]))  # obtenir le path du dir puis prendre la base du path -> le nom du dir seul
        if dirName == "palm"         : y = 0
        if dirName == "two"          : y = 1
        if dirName == "fist_moved"   : y = 2

        sample={
            'x': Image.open(self.data[index]),
            'y': y,               
            'fileName': os.path.basename(self.data[index]),
            'dirName': dirName
        }
        if self.transform is not None : sample = self.transform(sample)


        return sample

    @property
    def raw_folder(self) -> str:
        """
        Get the path of Raw folder
        """
        return os.path.join(CUR_DIR_PATH,'../Raw')

    @property
    def processed_folder(self) -> str:
        """
        Get the path of Processed folder
        """
        return os.path.join(CUR_DIR_PATH,'../Processed')
    
    def __len__(self) -> int:
        """
        Compute the number of image in the Raw folder
        """
        return len(self.data)

if __name__ == '__main__':
    print(CUR_DIR_PATH)
    #Check any problems in the code
    from torchvision.transforms import Compose
    from TransformHgr import MyOneHotEncoding, MyPilToTensor, MyResize, MyRandomHorizontalFlip, MyRandomRotation, MyRandomPerspective, MyRandomAffine, MyThreshold


    #Transforms
    TrToTensor = MyPilToTensor()
    TrOneHotEncoding = MyOneHotEncoding()
    TrResize = MyResize()
    TrRandomVerticalFlip=MyRandomHorizontalFlip()
    TrRandomRotation=MyRandomRotation()
    TrRandomPerspective= MyRandomPerspective()
    TrRandomAffine = MyRandomAffine()
    TrThreshold = MyThreshold()


    dataset= DataSetHgr(transform=Compose([TrToTensor,TrRandomAffine, TrResize,TrOneHotEncoding]))

                                
    print(len(dataset))
    i = 8
    print(dataset[0])           # identique à dataset.__getitem__(0)
    print(dataset[i]['x'].shape)    
    print(dataset[i]['y'])
    print(dataset[i]['fileName'])
    print(dataset[i]['dirName'])

    # # Histogramme d'une image

    img = dataset[i]['x']
    img = torch.squeeze(img)
    plt.figure()
    plt.imshow(img)
    plt.show()
    
    # plt.figure()
    # plt.hist(np.ravel(img.numpy()), bins = 256, density=True)
    # plt.show()

    # # Histogramme moyen du dataset

    # for k in range(1,len(dataset)):
    #     img = img + dataset[k]['x']
    
    # img = img/len(dataset)
    # plt.figure()
    # plt.hist(np.ravel(img.numpy()), bins = 256, density=True)
    # plt.show()