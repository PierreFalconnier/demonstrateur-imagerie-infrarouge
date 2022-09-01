import torch
from torch.functional import norm
from torch.nn import functional as F
from torch import nn
from pytorch_lightning import LightningModule
from math import gamma, isnan
from torch.optim import Adam,SGD
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR
from torchmetrics import ConfusionMatrix, Accuracy, Recall, Precision, F1Score
from torchvision.utils import make_grid
import numpy as np
from math import isnan
import matplotlib.pyplot as plt
import itertools
from typing import Any, Dict, List, Tuple, Type, Callable, Optional


class Cnn(LightningModule):
    """
    Attributs:
        hparams (Dict): Contains all Args oPrecisionf the constructor
        layer_1 (nn.Module): Contain a dense layer
        Loss (nn.Loss): Contain a Loss function
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        batch_size: int,
        bias: bool = True,
        learning_rate: float = 1e-4,
        Optimizer: Type[Optimizer] = SGD,
        Loss= nn.MSELoss,
        l1_coef: float = 0.0,
        l2_coef: float = 0.0,
        dropout: float = 0.0,
        # new args
        kernel_size: int=3,
        bnorm: bool=True,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.save_hyperparameters() #Needed if you want to load a trained model
        print(self.hparams.input_dim)
        

        ########################################
        ########## Convolutional Part ##########
        ########################################
        # Remind mnist images are (1, 28, 28) (channels, width, height)

        # if no padding : each conv we loose (Dim_kernel -1) pixels

        # instance first conv
        self.conv_1 = nn.Conv2d(
            in_channels=self.hparams.input_dim[0],
            out_channels= 16,
            kernel_size= self.hparams.kernel_size,
            stride= 1, #sample step of convolution
            padding=(self.hparams.kernel_size-1)//2 #default put 0
            )
            # on peut utiliser la string "same" (d'après la doc), ou avec des pading différents (Pstart, Pend)=( floor([S ceiling(I/S)-I+F-S]/2) , ceiling([S ceiling(I/S)-I+F-S]/2)     )
            # Choose padding : Dim_out = Dim_in + 2 x Dim_pad - Dim_kernel +1, if we want Dim_out=Dim_in -> Dim_pad = (Dim_kernel -1)/2. NB: Kernel size must be odd
            # (torch.floor(stride*torch.ceil((in_channels/stride)-in_channels+kernel_size-stride)/2),torch.ceil(stride*torch.ceil((in_channels/stride)-in_channels+kernel_size-stride)/2))
        # instance second conv
        self.conv_2 = nn.Conv2d(
            in_channels=16,
            out_channels= 32,
            kernel_size= self.hparams.kernel_size,
            stride= 1,
            padding=(self.hparams.kernel_size-1)//2
            )

        # instance bnorm
        if self.hparams.bnorm == False:
            self.bnorm_1 = lambda x : x # ne rien faire
            self.bnorm_2 = lambda x : x 
        else:
            self.bnorm_1 = nn.BatchNorm2d(16)
            self.bnorm_2 = nn.BatchNorm2d(32)

        self.maxpool = nn.MaxPool2d(2)              # image 28*28 to 14*14
        self.activation = nn.LeakyReLU(inplace=True)  # inplace pour éviter une copie, on utilisera donc self.activation(x) plutôt que x = self.activation(x)
        

        ########################################
        ######### Fully Connected Part #########
        ########################################
        # Instance Linear
        self.fc = nn.Linear(32*(self.hparams.input_dim[1]//2)*(self.hparams.input_dim[2]//2), 
                                self.hparams.num_classes, self.hparams.bias) # input should be a batch of vector
        # Instance dropout
        self.drop = nn.Dropout(p=self.hparams.dropout) 

        ########################################
        ########### Loss/Metric Part ###########
        ########################################
        self.Loss=self.hparams.Loss(reduction='mean')
        self.best_val_loss=float('nan')
        self.best_val_accuracy=float('nan')
        self.best_val_recall=float('nan')
        self.best_val_precision=float('nan')
        self.best_val_f1score = float('nan')
        self.confMat=ConfusionMatrix(num_classes=self.hparams.num_classes,normalize='true')
        self.accuracy = Accuracy()
        self.recall = Recall()
        self.precisionMetric = Precision()
        self.f1score = F1Score()
        # set at the moment this sample to None
        self.sample_image= None
        self.example_input_array= torch.rand((1,1,self.hparams.input_dim[1],self.hparams.input_dim[2]))    

    def forward(self, x: torch.Tensor)->torch.Tensor:
        """
        Forward method of our model.

        Args:
            x (torch.Tensor): input data
        
        Returns:
            y (torch.Tensor): Predicitons
        """
        # forward  input/output size : (b, 1, 640, 240) -> (b, 10)
        x = self.conv_1(x) #(b, 16, dim1, dim2)
        x = self.bnorm_1(x)
        self.activation(x)

        x = self.conv_2(x) #(b, 32, dim1, dim2)
        x = self.bnorm_2(x)
        self.activation(x)


        x = self.maxpool(x) #(b, 32, dim1/2, dim2/2)

        # x = torch.flatten(x, 1)
        # x = x.view(x.size(0), 32*dim1/2*dim1/2) #in_features = out_channels_conv x Dim_out_X x Dim_out_Y
        x = x.view(x.size(0), -1) # do the same
        x = self.drop(x)
        x = self.fc(x)
        # x = F.softmax(x, dim= 1)  # couche déjà implémentée dans la cross entropy de pytorch (soft max suivie de Nll)
        return x


    def training_step(self, batch: torch.Tensor, batch_idx: int)->dict:
        """
        Operates on a single batch of data from the training set with the forward function.
        
        Args:
            batch (torch.Tensor): A set of samples
            batch_idx (int): Index of the batch used for the step
        
        Returns:
            (dict): A dictionary. Can include any keys, but must include the key ``'loss'``
        
        Note:
            The loss value shown in the progress bar is smoothed (averaged) over the last values,
            so it differs from the actual loss returned in train/validation step.
            During the training, all tensor are by default set as required_grad as True.

        """
        #x: (batch_size, 768) y: (batch_size, nb_class)
        x, y = batch["x"],batch["y"]

        
        #y_hat: prediction (batch_size, nb_class)
        y_hat = self(x)

        #loss 
        loss = self.Loss(y_hat, y)

        # L1 regularisation
        if self.hparams.l1_coef > 0:
            l1_reg=0
            for name,parameters in self.named_parameters():
                if name[-4:]=='bias': continue #No need bias for regularisation
                elif name[:5]=='bnorm': continue #No need bnorm params for regularisation
                #You can chose the weights'layer to regul. with if name =='conv1.weight' : do as you want
                l1_reg += parameters.abs().sum()
            loss += self.hparams.l1_coef * l1_reg

        # L2 regularisation
        if self.hparams.l2_coef > 0:
            l2_reg=0
            for name,parameters in self.named_parameters():
                if name[-4:]=='bias': continue #No need bias for regularisation
                elif name[:5]=='bnorm': continue #No need bnorm params for regularisation
                l2_reg += parameters.pow(2).sum()
            loss += self.hparams.l2_coef * l2_reg
        
        self.log("Loss_each_step", {"Train" : loss }, on_step=True, logger=True)   # pour voir l'évolution de batch en batch
        
        return {'loss' : loss, 'x': x}

    def validation_step(self, batch: torch.Tensor, batch_idx: int)->dict:
        """
        Operates on a single batch of data from the validation set with the forward function.
        During the validation, all tensor are by default set as required_grad as False.

        Args:
            batch (torch.Tensor): A set of samples
            batch_idx (int): Index of the batch used for the step
        
        Returns:
            **Dictionary**, Can include any keys, but must include the key ``'loss'``

        Note:
            When the :meth:`validation_step` is called, the model has been put in eval mode
            and PyTorch gradients have been disabled. At the end of validation,
            the model goes back to training mode and gradients are enabled.
        """
        #x: (batch_size, 768) y: (batch_size, nb_class)
        x, y = batch["x"],batch["y"]
        

        #y_hat: prediction (batch_size, nb_class)
        y_hat = self(x)

        #loss 
        loss = self.Loss(y_hat, y)

        #y: target (batch_size, 1) 
        y= torch.argmax(y.to(torch.int),dim=1)

        #Metrics : Confusion matrix for one batch
        self.confMat(y_hat, y)
        accuracy = self.accuracy(torch.argmax(y_hat,dim=1),y)
        recall = self.recall(torch.argmax(y_hat,dim=1),y)
        precision = self.precisionMetric(torch.argmax(y_hat,dim=1),y)
        f1score = self.f1score(torch.argmax(y_hat,dim=1),y)


        # Logs
        self.log("Loss_each_step", {"Val" : loss }, on_step=True, logger=True)   # pour voir l'évolution de batch en batch
        self.log('val_loss',loss,logger=True) 

        # log val_loss every validation step  
        # It is needed if you want to use the callback ModelCheckpoint() which save the best top-k models.

        return {'val_loss' : loss ,'val_accuracy' : accuracy,'val_recall' : recall,'val_precision' : precision , 'val_f1score' : f1score}

    def configure_optimizers(self)->dict:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.

        Return:
            - **Dictionary**, with an ``"optimizer"`` key, and (optionally) a ``"lr_scheduler"``
              key whose value is a single LR scheduler or lr_dict.

        """
        if self.hparams.Optimizer == SGD:
            optim =  self.hparams.Optimizer(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum, nesterov=self.hparams.nesterov)
        if self.hparams.Optimizer == Adam:
            optim =  self.hparams.Optimizer(self.parameters(), lr=self.hparams.learning_rate)

        if self.hparams.Scheduler == OneCycleLR :
            scheduler = self.hparams.Scheduler(optim, max_lr=self.hparams.max_lr, total_steps=self.hparams.total_steps, pct_start=0.3,\
                div_factor= self.hparams.div_factor, final_div_factor=self.hparams.final_div_factor)
        if self.hparams.Scheduler == ExponentialLR :
            scheduler = self.hparams.Scheduler(optim, gamma = self.hparams.gamma)
        

        lr_dict = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": 'learning rate',
        }

        return {
           'optimizer': optim,
           'lr_scheduler': lr_dict
       }

    def training_epoch_end(self, training_step_outputs: List[Dict[str, torch.Tensor]]) -> None:
        """
        Called at the end of the training epoch with the outputs of all training steps.

        Args:
            training_step_outputs: List of outputs you defined in :meth:`training_step`, or if there are
                multiple dataloaders, a list containing a list of outputs for each dataloader.
        """
        #Compute the average train loss of one epoch
        train_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.logger.experiment.add_scalars('Loss_each_epoch',{'Train' : train_loss},self.current_epoch)
    
        self.log_weights_hist()

        # extract one sample_image
        if self.current_epoch == 1 :
            self.sample_image= torch.unsqueeze(training_step_outputs[0]['x'][0],dim=0) #We choose the first image of the epoch 1 as ref image
            #hint look training_step_outputs and remind that the sample should have size (b, channel, x, y)

        # call log_conv_1_filter and log_conv_2_filter
        if self.current_epoch>0 :
            out=self.log_conv_1_filter()
            self.log_conv_2_filter(out)


    def validation_epoch_end(self, validation_step_outputs: List[Dict[str, torch.Tensor]]) -> None:
        """
        Called at the end of the validation epoch with the outputs of all validation steps.

        Args:
            validation_step_outputs: List of outputs you defined in :meth:`validation_step`, or if there
                are multiple dataloaders, a list containing a list of outputs for each dataloader.
        """
        # Compute the average validation loss of one epoch
        val_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()

        self.logger.experiment.add_scalars('Loss_each_epoch',{'Val' : val_loss},self.current_epoch)

        if isnan(self.best_val_loss) | (float(val_loss.cpu().detach().numpy())<self.best_val_loss) :
            self.best_val_loss=float(val_loss.cpu().detach().numpy())
        
        # Compute confusion matrix over all validation batches and log it
        confMat= np.array(self.confMat.compute().detach().cpu().numpy())
        # fig = self.plot_confusion_matrix(confMat,["palm","l","fist","fist_moved","thumb","index","ok","palm_moved","c","down" ])
        fig = self.plot_confusion_matrix(confMat,["fist","palm","index","L" ])
        # fig = self.plot_confusion_matrix(confMat,["palm","two","fist"])
        self.logger.experiment.add_figure('0_confusion matrix',fig,self.current_epoch)
        
        # Compute the average accuracy of one epoch
        val_accuracy = torch.stack([x['val_accuracy'] for x in validation_step_outputs]).mean()
        self.logger.experiment.add_scalars('Accuracy',{'Val' : val_accuracy},self.current_epoch)
        # self.log('Accuracy',{'Val' : val_accuracy},on_epoch=True) 
        if isnan(self.best_val_accuracy) | (float(val_accuracy.cpu().detach().numpy())<self.best_val_accuracy) :
            self.best_val_accuracy=float(val_accuracy.cpu().detach().numpy())

        # Compute the average recall of one epoch
        val_recall = torch.stack([x['val_recall'] for x in validation_step_outputs]).mean()
        self.logger.experiment.add_scalars('Recall',{'Val' : val_recall},self.current_epoch)
        if isnan(self.best_val_recall) | (float(val_recall.cpu().detach().numpy())<self.best_val_recall) :
            self.best_val_recall=float(val_recall.cpu().detach().numpy())


        # Compute the average precision of one epoch
        val_precision = torch.stack([x['val_precision'] for x in validation_step_outputs]).mean()
        self.logger.experiment.add_scalars('Precision',{'Val' : val_precision},self.current_epoch)
        if isnan(self.best_val_precision) | (float(val_precision.cpu().detach().numpy())<self.best_val_precision) :
            self.best_val_precision=float(val_precision.cpu().detach().numpy())

        # Compute the average precision of one epoch
        val_f1score = torch.stack([x['val_f1score'] for x in validation_step_outputs]).mean()
        self.logger.experiment.add_scalars('F1score',{'Val' : val_f1score},self.current_epoch)
        if isnan(self.best_val_f1score) | (float(val_f1score.cpu().detach().numpy())<self.best_val_f1score) :
            self.best_val_f1score=float(val_f1score.cpu().detach().numpy())

    def on_train_epoch_end(self) :
        self.logger.experiment.flush() #Update tensorboard at every epoch (end)

    def log_weights_hist(self):
        #Iter all parameters and log it
        for name,parameters in self.named_parameters():
            self.logger.experiment.add_histogram(name,parameters,self.current_epoch)
            self.logger.experiment.add_histogram(name+'.grad',parameters.grad,self.current_epoch)#gradients

    
    def log_conv_1_filter(self):
        # apply first convolution on sample_image
        out=torch.squeeze(self.conv_1(self.sample_image)) # (1, 1, 28, 28) ->  (1, 32, 28, 28) -> (32, 28, 28)
        # make a grid image to log all features maps inside one figure : https://pytorch.org/vision/stable/utils.html 
        #image to display should be (B x C x H x W, hint: out.size() is (1, nb feature, x, y) but we want to log nb feature.
        out=torch.unsqueeze(out,dim=1) # (32, 28, 28) -> (32, 1, 28, 28)
        grid= make_grid(out.cpu().detach(),normalize=True) 
        self.logger.experiment.add_image('1_Conv_1 filter',grid,self.current_epoch)  # add the image to the logger 

        # apply first activation
        out=torch.squeeze(self.activation(self.bnorm_1(self.conv_1(self.sample_image))))
        grid= make_grid(torch.unsqueeze(out,dim=1).cpu().detach(),normalize=False) #image to display should be (B x C x H x W) 
        self.logger.experiment.add_image('2_activation_1',grid,self.current_epoch) 
        
        return out


    def log_conv_2_filter(self,input):
        #input correspond to the result of first conv layer -> batch normalisation -> leaky relu
        # apply second conv
        input=torch.unsqueeze(input,dim=0)
        out=torch.squeeze(self.conv_2(input))
        out=torch.unsqueeze(out,dim=1)
        grid= make_grid(out.cpu().detach(),normalize=True) #image to display should be (B x C x H x W) 
        self.logger.experiment.add_image('3_Conv_2 filter',grid,self.current_epoch)  

        out=torch.squeeze(self.activation(self.bnorm_2(self.conv_2(input))))
        out=torch.unsqueeze(out,dim=1)
        grid= make_grid(out.cpu().detach(),normalize=False) #image to display should be (B x C x H x W) 
        self.logger.experiment.add_image('4_activation_2',grid,self.current_epoch) 


    @staticmethod
    def plot_confusion_matrix(cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
            cm (array, shape = [n, n]): a confusion matrix of integer classes
            class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Compute the labels from the normalized confusion matrix.
        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

class Nll(nn.Module):
    def __init__(self,reduction: str= 'sum'):
        super().__init__()
        if reduction=='sum':
            self.Reduction=torch.sum
        if reduction=='mean':
            self.Reduction=torch.mean

    def forward(self,x,y):
        #Compute -log(prob)
        log_prob = -1.0 * torch.log(x)
        #exctract the error corresponding to the labels
        y=torch.argmax(y,dim=1)
        loss = log_prob.gather(dim=1, index=y.unsqueeze(1))
        #Compute the reduction
        loss= self.Reduction(loss)
        return loss

if __name__=='__main__':

    nll=Nll(reduction='mean')
    crossEntropy = nn.CrossEntropyLoss() #sofmax -> -log
    batch_size = 4
    # batch_size = 1024
    nb_classes = 10
    x = torch.randn(batch_size, nb_classes, requires_grad=True)
    y = torch.randint(0, nb_classes, (batch_size,nb_classes))
    loss_reference = crossEntropy(x, torch.argmax(y,dim=1))
    x= F.softmax(x, 1)
    loss = nll.forward(x, y)
    print(loss_reference - loss)
    print(x.shape)
    print(y.shape)