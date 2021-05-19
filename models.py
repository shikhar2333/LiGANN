
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchvision.models import resnet18

# define weight initialization
def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, kernel_size=3, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm3d(out_size, 0.8))
        layers.append(nn.LeakyReLU(0.2, True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_size, out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_size, 0.8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

'''
    Generator Model for BicycleGAN( based on U-net )
'''
class Generator(nn.Module):
    def __init__(self, latent_dim, voxel_array_shape):
        super(Generator, self).__init__()
        voxel_channels, self.h, self.w, self.d = voxel_array_shape
        self.reshape_z = nn.Linear(latent_dim, self.h*self.w*self.d)

        ## Downsampling Layers
        self.downsample1 = UNetDown(voxel_channels+1, 64, normalize=False)
        self.downsample2 = UNetDown(64, 128)
        self.downsample3 = UNetDown(128, 256)
        self.downsample4 = UNetDown(256, 512)
        self.downsample5 = UNetDown(512, 512)
        self.downsample6 = UNetDown(512, 512)
        self.downsample7 = UNetDown(512, 512, normalise=False)
        
        ## Transpose earlier kernels
        self.upsample1 = UNetUp(512, 512)
        self.upsample2 = UNetUp(1024, 512)
        self.upsample3 = UNetUp(1024, 512)
        self.upsample4 = UNetUp(1024, 256)
        self.upsample5 = UNetUp(512, 128)
        self.upsample6 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv3d(128, voxel_channels, kernel_size=3, stride=1, padding=1), nn.Tanh()
        ) 
    
    def forward(self, x, z):
        z = self.reshape_z(z).reshape(z.size(0), 1, self.h, self.w, self.d)
        d1 = self.downsample1( torch.cat((x, z), 1) )
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)
        d5 = self.downsample5(d4)
        d6 = self.downsample6(d5)
        d7 = self.downsample7(d6)
        u1 = self.upsample1(d7, d6)
        u2 = self.upsample2(u1, d5)
        u3 = self.upsample3(u2, d4)
        u4 = self.upsample4(u3, d3)
        u5 = self.upsample5(u4, d2)
        u6 = self.upsample6(u5, d1)

        return self.final(u6)

class Encoder(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(Encoder, self).__init__()
        resnet18_model = resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        self.pooling = nn.AvgPool3d(kernel_size=8, stride=8, padding=0)
        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar


class MultiDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm3d(out_filters, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        channels, _, _ = input_shape
        # Extracts discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv3d(512, 1, kernel_size=3, padding=1)
                ),
            )
        in_channels = 2
        self.downsample = nn.AvgPool3d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        # print("Input shape: ", x.shape)
        outs = self.forward(x)
        # for patch in outs:
        #     print("Patch Shape: ", patch.shape, patch[0][0][0])
        loss = sum([torch.mean((out - gt) ** 2) for out in outs])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            # print("Input shape fed to model: ", x.shape)
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs