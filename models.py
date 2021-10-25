
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn import init
from torch.nn.modules.conv import Conv3d
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

def Conv_Block_3D(input_channels, output_channels, normalize=True):
    layers = []
    layers.append(nn.Conv3d(input_channels, output_channels, kernel_size=4,
            stride=2, padding=1))
    if normalize:
        layers.append(nn.BatchNorm3d(output_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

def Conv_Block_3D_Transposed(input_channels, output_channels, normalize=True):
    layers = []
    layers.append(nn.ConvTranspose3d(input_channels, output_channels, kernel_size=4,
            stride=2, padding=1))
    if normalize:
        layers.append(nn.BatchNorm3d(output_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class Shape_VAE(nn.Module):
    """
    Returns autoencoded shape representation of the ligand
    """
    def __init__(self,in_channels=14) -> None:
        super().__init__()
        channels = in_channels
        # Encoder Model
        self.sequence1 = Conv_Block_3D(channels, 32)
        self.sequence2 = Conv_Block_3D(32, 64)
        self.sequence3 = Conv_Block_3D(64, 64)

        # return mu, log(sigma)
        self.fc1 = nn.Linear(64*3*3*3, 128)
        self.fc2 = nn.Linear(64*3*3*3, 128)

        # Decoder Model
        self.fc3 = nn.Linear(128, 64*3*3*3)

        self.sequence4 = Conv_Block_3D_Transposed(64, 64)
        self.sequence5 = Conv_Block_3D_Transposed(64, 32)
        self.sequence6 = Conv_Block_3D_Transposed(32, channels)

        # output 
        self.output = Conv3d(channels, channels, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x1 = self.sequence1(x)
        x2 = self.sequence2(x1)
        x3 = self.sequence3(x2)
        x4 = x3.view(x3.size(0), -1)
        # Returns mu and log(sigma)
        return self.fc1(x4), self.fc2(x4)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z1 = z.view(z.size(0), 64, 3, 3, 3)
        z2 = self.sequence4(z1)
        z3 = self.sequence5(z2)
        z4 = self.sequence6(z3)
        return z4

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        z1 = self.fc3(z)
        output = self.output(self.decode(z1))
        output1 = self.sigmoid(output)
        return output1, mu, sigma

    def loss(self, reconstructed_x, x, mu, logvar):
        BCE_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE_loss + KLD, BCE_loss, KLD

class CNN_Encoder(nn.Module):
    '''
    CNN Network which encodes the voxelised ligands into a vectorised form 
    '''
    def __init__(self, in_channels=14) -> None:
        super().__init__()
        channels = in_channels
        layers = []
        # Define the VGG-16 network

        # 2 conv layers followed by max pooling

        # First block
        layers.append(nn.Conv3d(channels, 32, padding=1, kernel_size=3,stride=1))
        layers.append(nn.BatchNorm3d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Conv3d(32, 32, padding=1,  kernel_size=3, stride=1))
        layers.append(nn.BatchNorm3d(32))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool3d(stride=2, kernel_size=2))

        # Second block
        layers.append(nn.Conv3d(32, 64, padding=1, kernel_size=3, stride=1))
        layers.append(nn.BatchNorm3d(64))
        layers.append(nn.ReLU())
        layers.append(nn.Conv3d(64, 64, padding=1, kernel_size=3, stride=1))
        layers.append(nn.BatchNorm3d(64))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool3d(stride=2, kernel_size=2))

        # Third block
        layers.append(nn.Conv3d(64, 128, padding=1, kernel_size=3, stride=1))
        layers.append(nn.BatchNorm3d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv3d(128, 128, padding=1, kernel_size=3, stride=1))
        layers.append(nn.BatchNorm3d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv3d(128, 128, padding=1, kernel_size=3, stride=1))
        layers.append(nn.BatchNorm3d(128))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool3d(stride=2, kernel_size=2))

#        layers.append(nn.Conv3d(128, 256, padding=1, kernel_size=3, stride=1))
#        layers.append(nn.BatchNorm3d(256))
#        layers.append(nn.ReLU())
#        layers.append(nn.Conv3d(256, 256, padding=1, kernel_size=3, stride=1))
#        layers.append(nn.BatchNorm3d(256))
#        layers.append(nn.ReLU())
#        layers.append(nn.Conv3d(256, 256, padding=1, kernel_size=3, stride=1))
#        layers.append(nn.BatchNorm3d(256))
#        layers.append(nn.ReLU())
#        layers.append(nn.MaxPool3d(stride=2, kernel_size=2))

        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(128, 256)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.detach().zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.features(x)
        print(x1.shape)
        x2 = x1.mean(dim=2).mean(dim=2).mean(dim=2)
        return self.fc(x2)

class EncoderCNN(nn.Module):
    def __init__(self, in_layers=14):
        super(EncoderCNN, self).__init__()
        self.pool = nn.MaxPool3d((2, 2, 2))
        self.relu = nn.ReLU()
        layers = []
        out_layers = 32

        for i in range(8):
            layers.append(nn.Conv3d(in_layers, out_layers, 3, bias=False, padding=1))
            layers.append(nn.BatchNorm3d(out_layers))
            layers.append(self.relu)
            in_layers = out_layers
            if (i + 1) % 2 == 0:
                # Duplicate number of layers every alternating layer.
                out_layers *= 2
                layers.append(self.pool)
        layers.pop()  # Remove the last max pooling layer!
        self.fc1 = nn.Linear(256, 512)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        x = x.mean(dim=2).mean(dim=2).mean(dim=2)
        x = self.relu(self.fc1(x))
        return x

class MolDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size=39, num_layers=1):
        super().__init__()
        self.embed_layer = nn.Embedding(vocab_size, embed_size)
        self.lstm_layer = nn.LSTM(embed_size, hidden_size, num_layers,
                batch_first=True)
        self.final_layer = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed_layer.weight.data.uniform_(-0.1, 0.1)
        self.final_layer.weight.data.uniform_(-0.1, 0.1)
        self.final_layer.bias.data.fill_(0)

    def forward(self, cnn_features, captions, lengths):
        embeddings = self.embed_layer(captions)
        embeddings = torch.cat((cnn_features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm_layer(packed)
        return self.final_layer(hiddens[0])

    def beam_search_sample(self, cnn_features, max_len=163):
        "Sample the smile representation given some encoded shape information"
        sampled_smile_idx = []
        inputs = cnn_features.unsqueeze(1)
        states = None
        for i in range(max_len):
            hidden_states, states = self.lstm_layer(inputs, states)
            squeezed_hidden = hidden_states.squeeze(1)
#            print(i,squeezed_hidden)
#            for i in range(5):
#                print(i,squeezed_hidden[i])
            outputs = self.final_layer(hidden_states.squeeze(1))
            predicted = outputs.max(1)[1]
            print(predicted)
            sampled_smile_idx.append(predicted)
            inputs = self.embed_layer(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_smile_idx
    def sample_prob(self, features, max_len=183, states=None):
        """Samples SMILES tockens for given shape features (probalistic picking)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(max_len):  # maximum sampling length
            hiddens, states = self.lstm_layer(inputs, states)
            outputs = self.final_layer(hiddens.squeeze(1))
            if i == 0:
                predicted = outputs.max(1)[1]
            else:
                probs = F.softmax(outputs, dim=1)

                # Probabilistic sample tokens
                if probs.is_cuda:
                    probs_np = probs.data.cpu().numpy()
                else:
                    probs_np = probs.data.numpy()

                rand_num = np.random.rand(probs_np.shape[0])
                iter_sum = np.zeros((probs_np.shape[0],))
                tokens = np.zeros(probs_np.shape[0], dtype=np.int)

                for i in range(probs_np.shape[1]):
                    c_element = probs_np[:, i]
                    iter_sum += c_element
                    valid_token = rand_num < iter_sum
                    update_indecies = np.logical_and(valid_token,
                                                     np.logical_not(tokens.astype(np.bool)))
                    tokens[update_indecies] = i

                # put back on the GPU.
                if probs.is_cuda:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)).cuda())
                else:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)))

            sampled_ids.append(predicted)
            inputs = self.embed_layer(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids



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

class Generator(nn.Module):
    '''
        Generator Model for BicycleGAN( based on U-net )
    '''
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

if __name__ == "__main__":
    encoder_model = CNN_Encoder()
    rand_tensor = torch.randn(1, 14, 24,24, 24)
    encoder_output = encoder_model(rand_tensor)
    print(encoder_output.shape)

