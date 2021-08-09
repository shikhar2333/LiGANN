#!/usr/bin/env python

import molgrid
import matplotlib.pyplot as plt
import argparse
from models import *
import torch.optim as optim
import sys

# set some constants
batch_size = 5
datadir = '/scratch/shubham/crossdock_data'
fname = datadir+"/custom_cd.types" 
cuda = True

molgrid.set_random_seed(0)
torch.manual_seed(0)

def extract_sdf_file(gninatypes_file):
    path = gninatypes_file.split("/")
    base_name = path[1].split(".")[0]
    base_name = base_name.rsplit("_", 1)[0]
    base_name += ".sdf"
    return datadir + "/structs/" + path[0] + "/" + base_name

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--lambda_pixel", type=float, default=10, help="pixelwise loss weight")
parser.add_argument("--lambda_latent", type=float, default=0.5, help="latent loss weight")
parser.add_argument("--lambda_kl", type=float, default=0.01, help="kullback-leibler loss weight")
opt = parser.parse_args()


# use the libmolgrid ExampleProvider to obtain shuffled, balanced, and stratified batches from a file
e = molgrid.ExampleProvider(data_root=datadir+"/structs",cache_structs=False, balanced=True,shuffle=True)
e.populate(fname)

ex = e.next()
c = ex.coord_sets[0]
center = tuple(c.center())

# initialize libmolgrid GridMaker
gmaker = molgrid.GridMaker()

# e.num_types()//2 is the number of channels used for voxel representation of docked ligand
print("Number of channels: ", e.num_types()//2)
dims = gmaker.grid_dimensions(e.num_types()//2)

mgridout = molgrid.MGrid4f(*dims)
gmaker.forward(center, c, mgridout.cpu())
molgrid.write_dx("tmp.dx", mgridout[0].cpu(), center, 0.5)

print("4D Tensor Shape: ", dims)
tensor_shape = (batch_size,)+dims
print(tensor_shape)
# molgrid.write_dx("temp",gmaker,center,1)


# Initialize Generator, Enocoder, VAE and LR Discriminator on GPU
generator = Generator(8, dims).to('cuda')
encoder = Encoder(vaeLike=True).to('cuda')
D_VAE = MultiDiscriminator(dims).to('cuda')
D_LR = MultiDiscriminator(dims).to('cuda')

# Initialise weights
generator.apply(weights_init)
D_VAE.apply(weights_init)
D_LR.apply(weights_init)

# construct optimizers for the 4 networks
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=[0.5, 0.999])
optimizer_E = optim.Adam(encoder.parameters(), lr=0.0002, betas=[0.5, 0.999])
optimizer_D_VAE = optim.Adam(D_VAE.parameters(), lr=0.0002, betas=[0.5, 0.999])
optimizer_D_LR = optim.Adam(D_LR.parameters(), lr=0.0002, betas=[0.5, 0.999])

# construct input tensors
input_tensor1 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
input_tensor2 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
float_labels = torch.zeros(batch_size, dtype=torch.float32)


# In[6]:


from torch.autograd import Variable
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), 8))))
    z = sampled_z * std + mu
    return z


# In[7]:


total_params = sum(p.numel() for p in generator.parameters())
train_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
print("Total Parameters: ", total_params)
print("Trainable Parameters: ", train_params)


# In[8]:


total_params = sum(p.numel() for p in D_VAE.parameters())
train_params = sum(p.numel() for p in D_VAE.parameters() if p.requires_grad)
print("Total Parameters: ", total_params)
print("Trainable Parameters: ", train_params)


# In[9]:


# Loss functions
mae_loss = torch.nn.L1Loss()
opt = {
  'lambda_pixel': 10,
   'lambda_kl': 0.01,
    'lambda_latent': 0.1
}
print(opt)


# In[ ]:


# train for 500 iterations
#G_loss, Pixel_loss, KL_Loss, Latent_loss, DVAE_loss, DLR_loss = [],[],[],[],[],[]
for iteration in range(500):
    # load data
    batch1 = e.next_batch(batch_size)
    batch2 = e.next_batch(batch_size)
    # libmolgrid can interoperate directly with Torch tensors, using views over the same memory.
    # internally, the libmolgrid GridMaker can use libmolgrid Transforms to apply random rotations and translations for data augmentation
    # the user may also use libmolgrid Transforms directly in python
    gmaker.forward(batch1, input_tensor1, 0, random_rotation=False)
    gmaker.forward(batch2, input_tensor2, 0, random_rotation=False)
    
    # Training the encoder and generator
    optimizer_E.zero_grad()
    optimizer_G.zero_grad()
    
    mu, logvar = encoder(input_tensor2)
    encoded_z = reparameterization(mu, logvar)

    fake_ligands = generator(input_tensor1, encoded_z)
    
    # L1 loss for measuring degree of diff between generated outputs and the actual input
    loss_pixel = mae_loss(fake_ligands, input_tensor2)
    
    # KL divergence between the distribution learned by the encoder and a random Gaussian
    loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
    
    # discrimantor distinguishing b/w fake and real for cVAE GAN
    loss_VAE_GAN = D_VAE.compute_loss(fake_ligands, 1)
    
    # sample z values from the Gaussian distribution with mu,sigma = 0, 1
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (input_tensor1.size(0), 8))))
    fake_ligands1 = generator(input_tensor1, sampled_z)
    
    # discrimantor distinguishing b/w fake and real for cLR GAN
    loss_LR_GAN = D_LR.compute_loss(fake_ligands1, 1)
    
    # Total Loss: Generator + Encoder
    loss_GE = loss_VAE_GAN + loss_LR_GAN + opt['lambda_pixel'] * loss_pixel + opt['lambda_kl'] * loss_kl
    
    loss_GE.backward(retain_graph=True)
    optimizer_E.step()
    
    # Generator only loss
    _mu, _ = encoder(fake_ligands1)
    loss_latent = opt['lambda_latent'] * mae_loss(_mu, sampled_z)
    
    loss_latent.backward()
    optimizer_G.step()
    
    # Train the discriminator for the cVAE GAN.
    optimizer_D_VAE.zero_grad()
    loss_D_VAE = D_VAE.compute_loss(fake_ligands.detach(), 0) + D_VAE.compute_loss(input_tensor2, 1)
    
    loss_D_VAE.backward()
    optimizer_D_VAE.step()
    
    # Train the discriminator for the cLR GAN.
    optimizer_D_LR.zero_grad()
    loss_D_LR = D_LR.compute_loss(fake_ligands1.detach(), 0) + D_LR.compute_loss(input_tensor2, 1)
    
    loss_D_LR.backward()
    optimizer_D_LR.step()
    
    sys.stdout.write(
            "\r[Epoch %d/%d] [G loss: %.3f, pixel loss: %.3f, kl loss: %.3f, latent loss: %.3f D_VAE loss: %.3f, D_LR loss: %.3f]"
            % (
                iteration,
                500,
                loss_GE.item(),
                loss_pixel.item(),
                loss_kl.item(),
                loss_latent.item(),
                loss_D_VAE.item(),
                loss_D_LR.item()
            )
    )

def plot(loss):
    plt.plot(loss)





