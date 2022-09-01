import os, sys
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR
from torch.optim import Adam,SGD
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.optim.optimizer import Optimizer


from Callback import MyCallBack

import matplotlib
matplotlib.use('Agg')

#include the path of the dataset(s) and the model(s)
CUR_DIR_PATH = Path(__file__).resolve()
ROOT = CUR_DIR_PATH.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))# add ROOT to PATH
CUR_DIR_PATH=os.path.dirname(CUR_DIR_PATH)

#include customed class
from Dataset.hgr.Dataloader.DataloaderHgr import DataloaderHgr
from Dataset.hgr.Dataloader.TransformHgr import MyOneHotEncoding, MyPilToTensor, MyResize, MyRandomHorizontalFlip, MyRandomRotation, MyRandomPerspective, MyRandomAffine
from Model.Cnn.Cnn import Cnn, Nll 
from torch import nn 


if __name__=='__main__':
    #Fix random
    pl.utilities.seed.seed_everything(42,True)

    #-----------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------
    #------------------all the hyperparmeters to fix (not in a parser to be more easy to use directly)---------------------
    #-----------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------
    
    #model
    kernel_size = 3

    #training
    batch_size = 72
    bnorm = True 
    epoch = 40
    
    #scheduler
    learning_rate_init = 0.001
    learning_rate_final = 0.0001
    max_lr= 0.005
    pct_start=0.2
    div_factor=max_lr/learning_rate_init
    final_div_factor= learning_rate_init/learning_rate_final

    #optimizer
    momentum=0.0 #0.9 # w(t+1) = w(t) - learning_rate.dL/dw(t) - mommentum.w(t-1)
    nesterov=False # w(t+1) = w(t) - learning_rate.dL/d(w(t) -  learning_rate.dL/dw(t)) - mommentum.w(t-1) the idea is to approximate the futur position

    #Regularisation
    l1_coef= 0.0
    # l1_coef= 0.02/512
    l2_coef= 0.0
    # l2_coef= 0.05/512
    dropout=0.2

    #-----------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------
    #-------------------------Transform and data augmetation ---------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------

    TrToTensor = MyPilToTensor()
    TrOneHotEncoding = MyOneHotEncoding()
    TrResize = MyResize()
    TrRandomHorizontalFlip=MyRandomHorizontalFlip()
    TrRandomRotation=MyRandomRotation()
    TrRandomPerspective= MyRandomPerspective()
    TrRandomAffine = MyRandomAffine()
    

    #-----------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------
    #-------------------------beginning of the training program-------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------


    #load dataset
    # dataloader = DataloaderHgr(batch_size= batch_size, transform = [TrResize, TrToTensor,TrOneHotEncoding])
    # dataloader = DataloaderHgr(batch_size= batch_size, transform = [TrToTensor, TrResize,TrOneHotEncoding])  # avec data augmenation
    dataloader = DataloaderHgr(batch_size= batch_size, transform = [TrToTensor, TrRandomHorizontalFlip, TrRandomAffine, TrResize,TrOneHotEncoding])  # avec data augmenation
    dataloader.prepare_data()
    dataloader.setup()

    #logger
    logger = TensorBoardLogger(os.path.join(CUR_DIR_PATH,'Log'),name='Cnn',default_hp_metric=False,log_graph=True)

    #callback
    myCallBack = MyCallBack()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Sauvegarde du meilleure modèle selon la val_loss
    checkpointCallback = ModelCheckpoint(
        dirpath=logger.log_dir,
        monitor='val_loss',
        save_top_k=1,
        verbose=False
        )#NB need to self.log in model val_loss in order to work

    #create a model
    model = Cnn(
        input_dim= dataloader.dims, #channel
        num_classes= len(dataloader.labels),
        batch_size=batch_size, #Not used but logged as hparams
        bias= True,
        learning_rate= learning_rate_init,
        # Optimizer= SGD,
        Optimizer= Adam,
        Loss= nn.CrossEntropyLoss,
        l1_coef= l1_coef,
        l2_coef= l2_coef,
        dropout= dropout,
        bnorm= bnorm,
        kernel_size= kernel_size,
        Scheduler= OneCycleLR,
        max_lr=max_lr,
        total_steps= epoch, #because learning_rate are updated at each epoch (option set in scheduler). 
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        momentum=momentum,
        nesterov=nesterov
        
        )

    model = Cnn(
        input_dim= dataloader.dims, #channel
        num_classes= len(dataloader.labels),
        batch_size=batch_size, #Not used but logged as hparams
        bias= True,
        learning_rate= learning_rate_init,
        # Optimizer= SGD,
        Optimizer= Adam,
        Loss= nn.CrossEntropyLoss,
        l1_coef= l1_coef,
        l2_coef= l2_coef,
        dropout= dropout,
        bnorm= bnorm,
        kernel_size= kernel_size,
        Scheduler= ExponentialLR,
        gamma = 0.9,
        momentum=momentum,
        nesterov=nesterov
        )

    #train the model
    trainer = pl.Trainer(max_epochs=epoch,gpus=1,logger=logger, callbacks=[myCallBack,lr_monitor, checkpointCallback], log_every_n_steps=1)
    trainer.fit(model, dataloader.train_dataloader(), dataloader.val_dataloader())