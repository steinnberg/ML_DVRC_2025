
# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                GAN / QuickDrawDataModule
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/MIAI - https://fidle.cnrs.fr
# ------------------------------------------------------------------
# JL Parouty (Mars 2024)



import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader


class QuickDrawDataModule(LightningDataModule):


    def __init__( self, dataset_file='./sheep.npy', scale=1., batch_size=64, num_workers=4 ):

        super().__init__()

        print('\n---- QuickDrawDataModule initialization ----------------------------')
        print(f'with : scale={scale}  batch size={batch_size}')
        
        self.scale        = scale
        self.dataset_file = dataset_file
        self.batch_size   = batch_size
        self.num_workers  = num_workers

        self.dims         = (28, 28, 1)
        self.num_classes  = 10



    def prepare_data(self):
        pass


    def setup(self, stage=None):
        print('\nDataModule Setup :')
        # Load dataset
        # Called at the beginning of each stage (train,val,test)
        # Here, whatever the stage value, we'll have only one set.
        data = np.load(self.dataset_file)
        print('Original dataset shape : ',data.shape)

        # Rescale
        n=int(self.scale*len(data))
        data = data[:n]
        print('Rescaled dataset shape : ',data.shape)

        # Normalize, reshape and shuffle
        data = data/255
        data = data.reshape(-1,28,28,1)
        data = torch.from_numpy(data).float()
        print('Final dataset shape    : ',data.shape)

        print('Dataset loaded and ready.')
        self.data_train = data


    def train_dataloader(self):
        # Note : Numpy ndarray is Dataset compliant
        # Have map-style interface. See https://pytorch.org/docs/stable/data.html 
        return DataLoader( self.data_train, batch_size=self.batch_size, num_workers=self.num_workers )