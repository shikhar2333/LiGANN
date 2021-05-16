# # The netwokrs are based on BycycleGAN (https://github.com/junyanz/BicycleGAN) and
# # Ligdream (https://github.com/compsciencelab/ligdream)
# # Networks are written in pytorch==0.3.1

import torch
import torch.nn as nn
import numpy as np

import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


class VAE(nn.Module):
    """
    Variational autoencoder for ligand shapes.
    This network is used only in training of the shape decoder.
    """
    def __init__(self, nc=5, ngf=128, ndf=128, latent_variable_size=512, use_cuda=False):
        super(VAE, self).__init__()
        self.use_cuda = use_cuda
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv3d(nc, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(32)

        self.e2 = nn.Conv3d(32, 32, 3, 2, 1)
        self.bn2 = nn.BatchNorm3d(32)

        self.e3 = nn.Conv3d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm3d(64)

        self.e4 = nn.Conv3d(64, ndf * 4, 3, 2, 1)
        self.bn4 = nn.BatchNorm3d(ndf * 4)

        self.e5 = nn.Conv3d(ndf * 4, ndf * 4, 3, 2, 1)
        self.bn5 = nn.BatchNorm3d(ndf * 4)

        self.fc1 = nn.Linear(512 * 3 * 3 * 3, latent_variable_size)
        self.fc2 = nn.Linear(512 * 3 * 3 * 3, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, 512 * 3 * 3 * 3)

        # up5
        self.d2 = nn.ConvTranspose3d(ndf * 4, ndf * 4, 3, 2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm3d(ndf * 4, 1.e-3)

        # up 4
        self.d3 = nn.ConvTranspose3d(ndf * 4, ndf * 2, 3, 2, padding=1, output_padding=1)
        self.bn7 = nn.BatchNorm3d(ndf * 2, 1.e-3)

        # up3 12 -> 12
        self.d4 = nn.Conv3d(ndf * 2, ndf, 3, 1, padding=1)
        self.bn8 = nn.BatchNorm3d(ndf, 1.e-3)

        # up2 12 -> 24
        self.d5 = nn.ConvTranspose3d(ndf, 32, 3, 2, padding=1, output_padding=1)
        self.bn9 = nn.BatchNorm3d(32, 1.e-3)

        # Output layer
        self.d6 = nn.Conv3d(32, nc, 3, 1, padding=1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, 512 * 3 * 3 * 3)
        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ndf * 4, 3, 3, 3)
        h2 = self.leakyrelu(self.bn6(self.d2((h1))))
        h3 = self.leakyrelu(self.bn7(self.d3(h2)))
        h4 = self.leakyrelu(self.bn8(self.d4(h3)))
        h5 = self.leakyrelu(self.bn9(self.d5(h4)))
        return self.sigmoid(self.d6(h5))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view())
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar


class EncoderCNN_v3(nn.Module):
    """
    CNN network to encode ligand shape into a vectorized representation.
    """
    def __init__(self, in_layers=5):
        super(EncoderCNN_v3, self).__init__()
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


class DecoderRNN(nn.Module):
    """
    RNN network to decode vectorized representation of a compound shape into SMILES string
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(62):  # maximum sampling length
            hiddens, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size),
            outputs = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, embed_size)
        return sampled_ids

    def sample_prob(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(62):  # maximum sampling length
            hiddens, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size),
            outputs = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
            if i == 0:
                predicted = outputs.max(1)[1]  # Do not start w/ different than start!
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
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids


class ListModule(object):
    # should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(
                self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class D_N3Dv1LayersMulti(nn.Module):
    """
    Disriminator network of 3D-BicycleGAN
    """
    def __init__(self, input_nc, ndf=64, norm_layer=None, use_sigmoid=False, gpu_ids=[], num_D=1):
        super(D_N3Dv1LayersMulti, self).__init__()

        self.gpu_ids = gpu_ids
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(
                input_nc, ndf, norm_layer, use_sigmoid)
            self.model = nn.Sequential(*layers)
        else:
            self.model = ListModule(self, 'model')
            layers = self.get_layers(
                input_nc, ndf, norm_layer, use_sigmoid)
            self.model.append(nn.Sequential(*layers))
            self.down = nn.AvgPool3d(3, stride=2, padding=[
                1, 1, 1], count_include_pad=False)
            for i in range(num_D - 1):
                ndf = int(round(ndf / (2 ** (i + 1))))
                layers = self.get_layers(
                    input_nc, ndf, norm_layer, use_sigmoid)
                self.model.append(nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d,
                   use_sigmoid=False):
        kw = 3
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw,
                              stride=1, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 5):
            use_stride = 1
            if n in [1, 3]:
                use_stride = 2

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=use_stride, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = 8
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        return sequence

    def parallel_forward(self, model, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(model, input, self.gpu_ids)
        else:
            return model(input)

    def forward(self, input):
        if self.num_D == 1:
            return self.parallel_forward(self.model, input)
        result = []
        down = input
        for i in range(self.num_D):
            result.append(self.parallel_forward(self.model[i], down))
            if i != self.num_D - 1:
                down = self.parallel_forward(self.down, down)
        return result


class G_Unet_add_all3D(nn.Module):
    """
    Generator network of 3D-BicycleGAN
    """
    def __init__(self, input_nc=7, output_nc=5, nz=8, num_downs=8, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False, gpu_ids=[], upsample='basic'):

        super(G_Unet_add_all3D, self).__init__()
        self.gpu_ids = gpu_ids
        self.nz = nz

        # construct unet blocks
        unet_block = UnetBlock_with_z3D(ngf * 8, ngf * 8, ngf * 8, nz, None, innermost=True,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
                                      stride=1)
        unet_block = UnetBlock_with_z3D(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout,
                                      upsample=upsample, stride=1)
        for i in range(num_downs - 6):  # 2 iterations
            if i == 0:
                stride = 2
            elif i == 1:
                stride = 1
            else:
                raise NotImplementedError("Too big, cannot handle!")

            unet_block = UnetBlock_with_z3D(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                            norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout,
                                            upsample=upsample, stride=stride)
        unet_block = UnetBlock_with_z3D(ngf * 4, ngf * 4, ngf * 8, nz, unet_block,
                                        norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
                                        stride=2)
        unet_block = UnetBlock_with_z3D(ngf * 2, ngf * 2, ngf * 4, nz, unet_block,
                                        norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
                                        stride=1)
        unet_block = UnetBlock_with_z3D(
            ngf, ngf, ngf * 2, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
            stride=2)
        unet_block = UnetBlock_with_z3D(input_nc, output_nc, ngf, nz, unet_block,
                                        outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
                                        stride=1)
        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)


class UnetBlock_with_z3D(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero',
                 stride=2):
        super(UnetBlock_with_z3D, self).__init__()
        
        downconv = []
        if padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        downconv += [nn.Conv3d(input_nc, inner_nc,
                               kernel_size=3, stride=stride, padding=p)]

        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type, stride=stride)
            down = downconv
            up = [uprelu] + upconv + [nn.Sigmoid()]
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type, stride=stride)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type, stride=stride)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3), x.size(4))
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero', stride=2):
    if upsample == 'basic':
        upconv = [nn.ConvTranspose3d(
            inplanes, outplanes, kernel_size=3, stride=stride, padding=1,
            output_padding=1 if stride == 2 else 0)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv


def fetch_simple_block3d(in_lay, out_lay, nl, norm_layer, stride=1, kw=3, padw=1):
    return [nn.Conv3d(in_lay, out_lay, kernel_size=kw,
                      stride=stride, padding=padw),
            nl(),
            norm_layer(out_lay)]


class E_3DNLayers(nn.Module):
    """
    E network of 3D-BicycleGAN
    """
    def __init__(self, input_nc=7, output_nc=5, ndf=64,
                 norm_layer='instance', nl_layer=None, gpu_ids=[], vaeLike=False):
        super(E_3DNLayers, self).__init__()
        self.gpu_ids = gpu_ids
        self.vaeLike = vaeLike

        nf_mult = 1
        kw, padw = 3, 1

        # Network
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw,
                              stride=1, padding=padw),
                    nl_layer()]
        # Repeats
        sequence.extend(fetch_simple_block3d(ndf, ndf * 2, nl=nl_layer, norm_layer=norm_layer))
        sequence.append(nn.AvgPool3d(2, 2))

        sequence.extend(fetch_simple_block3d(ndf * 2, ndf * 2, nl=nl_layer, norm_layer=norm_layer))
        sequence.append(nn.AvgPool3d(2, 2))

        sequence.extend(fetch_simple_block3d(ndf * 2, ndf * 4, nl=nl_layer, norm_layer=norm_layer))
        sequence.append(nn.AvgPool3d(2, 2))

        sequence.extend(fetch_simple_block3d(ndf * 4, ndf * 4, nl=nl_layer, norm_layer=norm_layer))

        sequence += [nn.AvgPool3d(3)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(ndf * 4, output_nc)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(ndf * 4, output_nc)])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)

        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output

