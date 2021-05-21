
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init
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
    def __init__(self, in_size, out_size, fraction=False):
        super(UNetUp, self).__init__()
#         scale = 1.5 if fraction else 2
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_size, out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_size, 0.8),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, skip_input):
        # print("old x shape: ", x.shape) 
        x = self.model(x)
        skip_input = self.upsample(skip_input)
        padding = (x.shape[2] - skip_input.shape[2])//2
        p3d = (padding, padding, padding, padding, padding, padding)
        skip_input = F.pad(skip_input, p3d, "constant", 0)
        # print(x.shape, skip_input.shape)
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
        self.downsample1 = UNetDown(voxel_channels+1, 32, normalize=False)
        self.downsample2 = UNetDown(32, 64)
        self.downsample3 = UNetDown(64, 64)
        self.downsample4 = UNetDown(64, 128)
#         self.downsample5 = UNetDown(128, 128)
#         self.downsample6 = UNetDown(128, 128, normalize=False)
#         self.downsample7 = UNetDown(128, 128, normalize=False)
        
        ## Transpose earlier kernels
        self.upsample1 = UNetUp(128, 64, fraction=True)
        self.upsample2 = UNetUp(128, 64)
        self.upsample3 = UNetUp(128, 64)
#         self.upsample4 = UNetUp(128, 32)
#         self.upsample5 = UNetUp(128, 64)
#         self.upsample6 = UNetUp(128, 32)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv3d(96, voxel_channels, kernel_size=3, stride=1, padding=1), nn.Tanh()
        ) 
    
    def forward(self, x, z):
        z = self.reshape_z(z).view(z.size(0), 1, self.h, self.w, self.d)
        # print("Z shape: ", z.shape)
        d1 = self.downsample1( torch.cat((x, z), 1) )
#         print("d1 shape: ", d1.shape)
        d2 = self.downsample2(d1)
#         print("d2 shape: ", d2.shape)
        d3 = self.downsample3(d2)
#         print("d3 shape: ", d3.shape)
        d4 = self.downsample4(d3)
#         print("d4 shape: ", d4.shape)
#         d5 = self.downsample5(d4)
#         print("d5 shape: ", d5.shape)
#         d6 = self.downsample6(d5)
        # print("d6 shape: ", d6.shape)
#         d7 = self.downsample7(d6)
        # print("d7 shape: ", d7.shape)

        u1 = self.upsample1(d4, d3)
#         print("u1 shape: ",u1.shape)
        u2 = self.upsample2(u1, d2)
#         print("u2 shape: ", u2.shape)
        u3 = self.upsample3(u2, d1)
#         print("u3 shape: ", u3.shape)
#         u4 = self.upsample4(u3, d1)
#         print("u4 shape: ", u4.shape)
#         u5 = self.upsample5(u4, d2)
#         u6 = self.upsample6(u5, d1)
#         print("u4 shape: ", u4.shape)
        return self.final(u3)
#         return 0


# class Encoder(nn.Module):
#     def __init__(self, latent_dim, input_shape):
#         super(Encoder, self).__init__()
#         resnet18_model = resnet18(pretrained=False)
#         self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
#         self.pooling = nn.AvgPool3d(kernel_size=8, stride=8, padding=0)
#         # Output is mu and log(var) for reparameterization trick used in VAEs
#         self.fc_mu = nn.Linear(256, latent_dim)
#         self.fc_logvar = nn.Linear(256, latent_dim)

#     def forward(self, img):
#         out = self.feature_extractor(img)
#         out = self.pooling(out)
#         out = out.view(out.size(0), -1)
#         mu = self.fc_mu(out)
#         logvar = self.fc_logvar(out)
#         return mu, logvar

def fetch_simple_block3d(in_lay, out_lay, nl, norm_layer, stride=1, kw=3, padw=1):
    return [nn.Conv3d(in_lay, out_lay, kernel_size=kw, stride=stride, padding=padw)]

class Encoder(nn.Module):
    """
    E network of 3D-BicycleGAN
    """
    def __init__(self, input_nc=14, output_nc=8, ndf=64,
                 norm_layer='instance', nl_layer=None, gpu_ids=[], vaeLike=False):
        super(Encoder, self).__init__()
        self.gpu_ids = gpu_ids
        self.vaeLike = vaeLike

        nf_mult = 1
        kw, padw = 3, 1

        # Network
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw,
                              stride=1, padding=padw)]
        # Repeats
        sequence.extend(fetch_simple_block3d(ndf, ndf * 2, nl=nl_layer, norm_layer=norm_layer))
        sequence.append(nn.AvgPool3d(2, 2))

        sequence.extend(fetch_simple_block3d(ndf * 2, ndf * 2, nl=nl_layer, norm_layer=norm_layer))
        sequence.append(nn.AvgPool3d(2, 2))

        sequence.extend(fetch_simple_block3d(ndf * 2, ndf * 4, nl=nl_layer, norm_layer=norm_layer))
        sequence.append(nn.AvgPool3d(2, 2))

        sequence.extend(fetch_simple_block3d(ndf * 4, ndf * 4, nl=nl_layer, norm_layer=norm_layer))
        sequence += [nn.AvgPool3d(kernel_size=3,stride=2)]
        sequence += [nn.AvgPool3d(kernel_size=2)]
        
#         sequence.extend(fetch_simple_block3d(ndf * 4, ndf * 4, nl=nl_layer, norm_layer=norm_layer))
#         sequence += [nn.AvgPool3d(kernel_size=3,stride=2)]
        
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(ndf * 4, output_nc)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(ndf * 4, output_nc)])

    def forward(self, x):
        x_conv = self.conv(x)
        # print("x_conv shape: ", x_conv.shape)
        conv_flat = x_conv.view(x.size(0), -1)

        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output


class MultiDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, kernel_size=3, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm3d(out_filters, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers
        # print(input_shape)
        channels, _, _, _ = input_shape
        # Extracts discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(channels, 32, normalize=False),
                    *discriminator_block(32, 64),
                    *discriminator_block(64, 128),
#                     *discriminator_block(128, 256),
                    nn.Conv3d(128, 1, kernel_size=3, padding=1)
                ),
            )
        in_channels = 2
        self.downsample = nn.AvgPool3d(in_channels, stride=2, padding=1, count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        # print("Input shape: ", x.shape)
        outs = self.forward(x)
        for patch in outs:
        #     print("Patch Shape: ", patch.shape, patch[0][0][0])
            loss = sum([torch.mean((out - gt) ** 2) for out in outs])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
#             print("Input shape fed to model: ", x.shape)
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs