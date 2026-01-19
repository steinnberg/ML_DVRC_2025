
# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                Basic GAN LigthningModule
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/MIAI - https://fidle.cnrs.fr
# ------------------------------------------------------------------
# JL Parouty (Mars 2024)



import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from lightning import LightningModule


class GAN(LightningModule):

    # -------------------------------------------------------------------------
    # Init
    # -------------------------------------------------------------------------
    #
    def __init__(
        self,
        data_shape          = (None,None,None),
        latent_dim          = None,
        lr                  = 0.0002,
        b1                  = 0.5,
        b2                  = 0.999,
        batch_size          = 64,
        generator_name      = None,
        discriminator_name  = None,
        **kwargs,
    ):
        super().__init__()

        print('\n---- GAN initialization --------------------------------------------')

        # ---- Hyperparameters
        #
        # Enable Lightning to store all the provided arguments under the self.hparams attribute.
        # These hyperparameters will also be stored within the model checkpoint.
        #
        self.save_hyperparameters()

        print('Hyperarameters are :')
        for name,value in self.hparams.items():
            print(f'{name:24s} : {value}')

        # ---- Because we have more than one optimizer
        #
        self.automatic_optimization = False

        # ---- Generator/Discriminator instantiation
        #
        print('Submodels :')
        module=sys.modules['__main__']
        class_g = getattr(module, generator_name)
        class_d = getattr(module, discriminator_name)
        self.generator     = class_g( latent_dim=latent_dim, data_shape=data_shape)
        self.discriminator = class_d( latent_dim=latent_dim, data_shape=data_shape)

        # ---- Validation and example data
        #
        self.validation_z        = torch.randn(8, self.hparams.latent_dim)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)


    def forward(self, z):
        return self.generator(z)


    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)


    def training_step(self, batch, batch_idx):
        imgs       = batch
        batch_size = batch.size(0)

        optimizer_g, optimizer_d = self.optimizers()

        # ---- Get some latent space vectors
        #      We use type_as() to make sure we initialize z on the right device (GPU/CPU).
        #
        z = torch.randn(batch_size, self.hparams.latent_dim)
        z = z.type_as(imgs)

        #============GENERATOR======================================================================
        # ---- Train generator ------------------------------------------------
        #      Generator use optimizer #0
        #      We try to generate false images that could mislead the discriminator
        #
        self.toggle_optimizer(optimizer_g)
        
        # Generate fake images
        self.fake_imgs = self.generator.forward(z)

        # Assemble labels that say all images are real, yes it's a lie ;-)
        # put on GPU because we created this tensor inside training_loop
        misleading_labels = torch.ones(batch_size, 1)
        misleading_labels = misleading_labels.type_as(imgs)

        # Adversarial loss is binary cross-entropy
        y_hat = self.discriminator.forward(self.fake_imgs)
        # print(y_hat)
        g_loss = self.adversarial_loss(self.discriminator.forward(self.fake_imgs), misleading_labels)
        
        self.log("g_loss", g_loss, prog_bar=True)
        
        # Backward loss
        self.manual_backward(g_loss)
        
        optimizer_g.step()
        optimizer_g.zero_grad()
        
        self.untoggle_optimizer(optimizer_g)
        
        #=================DISCRIMINATOR==========================================================
        # ---- Train discriminator --------------------------------------------
        #      Discriminator use optimizer #1
        #      We try to make the difference between fake images and real ones 
        #
        self.toggle_optimizer(optimizer_d)

        # These images are reals
        real_labels = torch.ones(batch_size, 1)
        real_labels = real_labels.type_as(imgs)
        pred_labels = self.discriminator.forward(imgs)

        real_loss   = self.adversarial_loss(pred_labels, real_labels)

        # These images are fake
        fake_imgs   = self.generator.forward(z)
        fake_labels = torch.zeros(batch_size, 1)
        fake_labels = fake_labels.type_as(imgs)

        fake_loss   = self.adversarial_loss(self.discriminator(fake_imgs.detach()), fake_labels)

        # Discriminator loss is the average
        d_loss = (real_loss + fake_loss) / 2
        
        self.log("d_loss", d_loss, prog_bar=True)
 
        # Backward
        self.manual_backward(d_loss)
        
        optimizer_d.step()
        optimizer_d.zero_grad()
        
        self.untoggle_optimizer(optimizer_d)
 


    def configure_optimizers(self):

        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        # With a GAN, we need 2 separate optimizer.
        # opt_g = torch.optim.Adam(self.generator.parameters(),     lr=lr, betas=(b1, b2))
        # opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2),)
        opt_g = torch.optim.Adam(self.generator.parameters(),     lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []


    def on_train_epoch_end(self):

        # ---- Log Graph
        #
        if(self.current_epoch==1):
            sampleImg=torch.rand((1,28,28,1))
            sampleImg=sampleImg.type_as(self.generator.model[0].weight)
            self.logger.experiment.add_graph(self.discriminator,sampleImg)

        # ---- Log some of these images
        #
        z = torch.randn(self.hparams.batch_size, self.hparams.latent_dim)
        z = z.type_as(self.generator.model[0].weight)
        sample_imgs = self.generator(z)
        sample_imgs = sample_imgs.permute(0, 3, 1, 2) # from NHWC to NCHW
        grid = torchvision.utils.make_grid(tensor=sample_imgs, nrow=12, )
        self.logger.experiment.add_image(f"Generated images", grid,self.current_epoch)
