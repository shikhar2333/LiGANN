
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn import init
from torch.nn.modules.conv import Conv3d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def loss(self, reconstructed_x, x, mu, logvar):
        BCE_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='mean')
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE_loss + KLD, BCE_loss, KLD

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
    def __init__(self,in_channels=5) -> None:
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
#        self.fc = nn.Linear(128, 256)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.detach().zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.features(x)
        x1 = x1.permute(0, 2, 3, 4, 1)
#        x2 = x1.mean(dim=2).mean(dim=2).mean(dim=2)
        return x1

class AttentionBlock(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim) -> None:
        """
        Attention Network
        """
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, encoder_output, decoder_hidden_state):
        encoder_att = self.encoder_att(encoder_output)
        decoder_att = self.decoder_att(decoder_hidden_state)
        attention_add = encoder_att + decoder_att.unsqueeze(1)
        attention_add = self.relu(attention_add)
        full_att = self.full_att(attention_add).squeeze(2)
        alphas = self.softmax(full_att)
        att_weighted_encoding = (encoder_output * alphas.unsqueeze(2)).sum(dim=1)
        return att_weighted_encoding, alphas

class MolDecoder(nn.Module):
    def __init__(self, embed_size, decoder_dim, att_dim,
            vocab_size=107, dropout=0.5, encoder_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention_block = AttentionBlock(encoder_dim, decoder_dim, att_dim)
        self.embed_layer = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim,
                bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.final_layer = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed_layer.weight.data.uniform_(-0.1, 0.1)
        self.final_layer.weight.data.uniform_(-0.1, 0.1)
        self.final_layer.bias.data.fill_(0)

    def init_hidden_states(self, encoder_output):
        mean_encoder_output = encoder_output.mean(dim=1)
        h = self.init_h(mean_encoder_output)
        c = self.init_c(mean_encoder_output)
        return h,c

    def forward(self,encoder_output, captions, caption_lengths):
        batch_size = encoder_output.shape[0]
        encoder_dim = encoder_output.shape[-1]
        encoder_output = encoder_output.view(batch_size, -1, encoder_dim)
        total_voxels = encoder_output.shape[1]

        embeddings = self.embed_layer(captions)
        h,c = self.init_hidden_states(encoder_output)

        decode_lengths = caption_lengths

        predictions = torch.zeros(batch_size, max(decode_lengths),
                self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths),
                total_voxels).to(device)
        for t_i in range(max(decode_lengths)):
            eff_batch_size_t_i = sum([l > t_i for l in decode_lengths])
            att_weighted_encoding, alpha = self.attention_block(encoder_output[:eff_batch_size_t_i],
                    h[:eff_batch_size_t_i])
            gate = self.sigmoid(self.f_beta(h[:eff_batch_size_t_i]))
            att_weighted_encoding = gate * att_weighted_encoding
            h, c = self.lstm_cell( torch.cat([embeddings[:eff_batch_size_t_i,
                t_i, :], att_weighted_encoding], dim=1),
                (h[:eff_batch_size_t_i], c[:eff_batch_size_t_i]) )
            pred = self.final_layer(self.dropout(h))
            predictions[:eff_batch_size_t_i, t_i, :] = pred
            alphas[:eff_batch_size_t_i, t_i, :] = alpha
        return predictions, alphas, decode_lengths

# define weight initialization
def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)

if __name__ == "__main__":
    encoder_model = CNN_Encoder()
    rand_tensor = torch.randn(1, 14, 24,24, 24)
    encoder_output = encoder_model(rand_tensor)
    print(encoder_output.shape)

