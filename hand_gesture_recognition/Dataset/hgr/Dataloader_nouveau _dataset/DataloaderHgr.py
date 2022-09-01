import os, sys
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from typing import Any, Dict, List, Tuple, Type, Callable, Optional

#include the path of the dataset(s)
CUR_DIR_PATH=os.path.dirname(__file__)
sys.path.append(CUR_DIR_PATH)
from DataSetHgr import DataSetHgr

class DataloaderHgr(pl.LightningDataModule):
    """
    A DataLoader ORGANIZES the data as training, validation, test set. It creates a batch of samples provide by a Dataset.

    Attributs:
        batch_size (int): The number of samples used in a training/validation step in a batch
        transform (list): Contains all the transforms will be applied on each samples
        fractionTrain (float): The fractions of samples used for the training
        fractionVal (float): The fractions of samples used for the validation
        fractionTest (float): The fractions of samples used for the test/inference
        dataset_train (Dataset): Dataset used for training
        dataset_val (Dataset): Dataset used for validation
        dataset_test (Dataset): Dataset used for test/inference
    """
    def __init__(self,batch_size: int = 1, transform: Optional[Callable] = None)-> None:
        """
            Constructor

            Args:
                batch_size (int): The number of samples used in a training/validation step in a batch
                transform (list): Contains all the transforms will be applied on each samples
        """
        super().__init__()
        self.transform = transforms.Compose(transform)
        self.batch_size=batch_size
        self.fractionTrain= None
        self.fractionVal= None
        self.fractionTest= None
        self.dataset= None
        self.train_dataset = None
        self.val_dataset= None
        self.test_dataset= None
        
    def prepare_data(self, fractionTrain: float= 0.7, fractionVal: float= 0.2, fractionTest: float= 0.1)-> None:
        """
        Method that updates the different fractions

        Args:
            fractionTrain (float): The fractions of samples used for the training
            fractionVal (float): The fractions of samples used for the validation
            fractionTest (float): The fractions of samples used for the test/inference
        """
        self.fractionTrain= fractionTrain
        self.fractionVal= fractionVal
        self.fractionTest= fractionTest
        if round((self.fractionTrain+self.fractionVal+self.fractionTest),5)!=1.0 : raise ValueError("Sum of train, val, test should be 1.0")
        

    def setup(self)-> None:
        """
        Method that set full/train/val/test datasets used by their respective dataloaders
        """
        # Assign full dataset
        self.dataset= DataSetHgr(transform=self.transform)
        #Compute nb sample for train and val
        nbSampleTotal=len(self.dataset)
        nbSampleTrain = round(nbSampleTotal*self.fractionTrain)
        nbSampleVal = round(nbSampleTotal*self.fractionVal)
        nbSampleTest = nbSampleTotal-nbSampleTrain-nbSampleVal
        # Assign val/test datasets for use in dataloaders
        self.train_dataset, self.val_dataset, self.test_dataset= random_split(self.dataset, [nbSampleTrain, nbSampleVal, nbSampleTest])

    
    def train_dataloader(self)->Type[DataLoader]:
        """
        Methode that return the training DataLoader

        Note:
            - batch_size correspond to the number of sample to be processed at each step (parrallel computation).
            - num_workers correspond to the number of subprocess created to feed the model. In general, I use N subprocess when my CPU have N thread.
              The efficiency depends if your CPU is busy or not during the training. You can also pass as hyperparameter, if you want.
            - pin_memory define the way you store the data in the RAM. If you want use a GPU, set it to True.
              NB: the data memory pipe is "read data -> pageable memory -> pinned memory -> GPU memory". If you set pin_memory to True, you accelerate the process but only if you use GPU.
              https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
            - shuffle correspond to a flag for data shuffling.
        """

        return DataLoader(self.train_dataset,batch_size=self.batch_size,num_workers=12,pin_memory=True,shuffle=True)

    def val_dataloader(self)->Type[DataLoader]:
        """
        Methode that return the validation DataLoader

         Note:
            - batch_size correspond to the number of sample to be processed at each step (parrallel computation).
            - num_workers correspond to the number of subprocess created to feed the model. In general, I use N subprocess when my CPU have N thread.
              The efficiency depends if your CPU is busy or not during the training. You can also pass as hyperparameter, if you want.
            - pin_memory define the way you store the data in the RAM. If you want use a GPU, set it to True.
              NB: the data memory pipe is "read data -> pageable memory -> pinned memory -> GPU memory". If you set pin_memory to True, you accelerate the process but only if you use GPU.
              https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
            - shuffle correspond to a flag for data shuffling.
        """
        return DataLoader(self.val_dataset,batch_size=self.batch_size,num_workers=1,pin_memory=True,shuffle=False)

    #No need test_dataloader for this toys exemple.

    @property
    def dims(self)->Tuple[int]:
        """
        Get the dimension of an image as (Nb channels, Nb rows, Nb cols)
        """
        sample  = self.train_dataset.__getitem__(0)
        return sample["x"].shape
    
    @property
    def labels(self)-> List[str]:
        """
        Get the labels as a list of string
        """
        return ["palm","two","fist"]
        # return ["fist","palm","index","L" ]
    

if __name__=='__main__':
    #Check any problems in the code
    from TransformHgr import MyOneHotEncoding, MyPilToTensor, MyResize

    #Fix random
    pl.utilities.seed.seed_everything(42,True)

    #Transforms
    TrToTensor = MyPilToTensor()
    TrOneHotEncoding = MyOneHotEncoding()
    TrResize = MyResize()

    dataloader = DataloaderHgr(transform = [TrResize, TrToTensor,TrOneHotEncoding])
    dataloader.prepare_data()
    dataloader.setup()
    sample= dataloader.train_dataset.__getitem__(0)
    # sample= dataloader.train_dataset[0]
    
    print(sample['x'])
    print(sample['x'].device)
    print(sample['y'])
    print(sample['fileName'])
    print(sample['x'].size())
    

    sample= next(iter(dataloader.train_dataloader()))
    print(sample['x'])
    print(sample['x'].device)
    print(sample['y'])
    print(sample['fileName'])
    print(sample['x'].size())

    print(len(dataloader.train_dataset))
    print(len(dataloader.train_dataloader()))
    print(dataloader.labels)
    print(dataloader.dims)