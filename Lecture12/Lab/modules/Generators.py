
# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                         GAN / Generators
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/MIAI - https://fidle.cnrs.fr
# ------------------------------------------------------------------
# JL Parouty (Mars 2024)



import numpy as np
import torch.nn as nn


# -----------------------------------------------------------------------------
# -- Generator n°1
# -----------------------------------------------------------------------------
#
class Generator_1(nn.Module):

    def __init__(self, latent_dim=None, data_shape=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape  = data_shape
        print('init generator 1         : ',latent_dim,' to ',data_shape)

        self.model = nn.Sequential(
            
            nn.Linear(latent_dim, 128),
            nn.ReLU(),

            nn.Linear(128,256),
            nn.BatchNorm1d(256, 0.8),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.ReLU(),

            nn.Linear(1024, int(np.prod(data_shape))),
            nn.Sigmoid()

        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


# -----------------------------------------------------------------------------
# -- Generator n°1
# -----------------------------------------------------------------------------
#
class Generator_2(nn.Module):

    def __init__(self, latent_dim=None, data_shape=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape  = data_shape
        print('init generator 2         : ',latent_dim,' to ',data_shape)

        self.model = nn.Sequential(
            
            nn.Linear(latent_dim, 7*7*64),
            nn.Unflatten(1, (64,7,7)),
            
            # nn.UpsamplingNearest2d( scale_factor=2 ),
            nn.UpsamplingBilinear2d( scale_factor=2 ),
            nn.Conv2d( 64,128, (3,3), stride=(1,1), padding=(1,1) ),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            # nn.UpsamplingNearest2d( scale_factor=2 ),
            nn.UpsamplingBilinear2d( scale_factor=2 ),
            nn.Conv2d( 128,256, (3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d( 256,1, (5,5), stride=(1,1), padding=(2,2)),
            nn.Sigmoid()

        )

    def forward(self, z):
        img_nchw = self.model(z)
        img_nhwc = img_nchw.permute(0, 2, 3, 1) # reformat from NCHW to NHWC
        # img = img.view(img.size(0), *self.img_shape) # reformat from NCHW to NHWC
        return img_nhwc



