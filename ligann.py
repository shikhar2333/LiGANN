#!/usr/bin/env python
# coding: utf-8

# In[70]:


import molgrid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import os
import matplotlib.pyplot as plt
import argparse


# In[71]:


# set some constants
batch_size = 50
datadir = '/scratch/shubham/crossdock_data'
fname = datadir+"/custom_cd.types" 

molgrid.set_random_seed(0)
torch.manual_seed(0)
np.random.seed(0)

'''
    Arguement Parsing
'''
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
# parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
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


# In[72]:


# define network architecture
# class Net(nn.Module):
#     def __init__(self, dims):
#         super(Net, self).__init__()
#         self.pool0 = nn.MaxPool3d(2)
#         self.conv1 = nn.Conv3d(dims[0], 32, kernel_size=3, padding=1)
#         self.pool1 = nn.MaxPool3d(2)
#         self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
#         self.pool2 = nn.MaxPool3d(2)
#         self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)

#         self.last_layer_size = dims[1]//8 * dims[2]//8 * dims[3]//8 * 128
#         self.fc1 = nn.Linear(self.last_layer_size, 2)

#     def forward(self, x):
#         x = self.pool0(x)
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = F.relu(self.conv3(x))
#         x = x.view(-1, self.last_layer_size)
#         x = self.fc1(x)
#         return x


# In[73]:


# In[74]:


# use the libmolgrid ExampleProvider to obtain shuffled, balanced, and stratified batches from a file
e = molgrid.ExampleProvider(data_root=datadir+"/structs",balanced=True,shuffle=True)
e.populate(fname)


# In[75]:


ex = e.next()
print(ex.coord_sets[0])
c = ex.coord_sets[0]
center = tuple(c.center())


# In[88]:


# initialize libmolgrid GridMaker
gmaker = molgrid.GridMaker()
# e.num_types() is the number of channels for voxel representation
# print("Number of channels: ", e.num_types())
dims = gmaker.grid_dimensions(e.num_types()//2)

mgridout = molgrid.MGrid4f(*dims)
gmaker.forward(center, c, mgridout.cpu())
molgrid.write_dx("tmp.dx", mgridout[0].cpu(), center, 0.5)

print("4D Tensor Shape: ", dims)
tensor_shape = (batch_size,)+dims
print(tensor_shape)
# molgrid.write_dx("temp",gmaker,center,1)


# In[89]:


# initialize Net on GPU
model = Net(dims).to('cuda')
model.apply(weights_init)


# In[90]:


# construct optimizer
optimizer_G = optim.Adam(model.parameters(), lr=opt.lr, momentum=0.9, betas=[opt.b1, opt.b2])


# In[91]:


# construct input tensors
input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
# print(input_tensor.shape)
float_labels = torch.zeros(batch_size, dtype=torch.float32)


# In[92]:


# train for 500 iterations
losses = []
for iteration in range(500):
    # load data
    batch = e.next_batch(batch_size)
    # libmolgrid can interoperate directly with Torch tensors, using views over the same memory.
    # internally, the libmolgrid GridMaker can use libmolgrid Transforms to apply random rotations and translations for data augmentation
    # the user may also use libmolgrid Transforms directly in python
    gmaker.forward(batch, input_tensor, 0, random_rotation=False)
    batch.extract_label(0, float_labels)
    labels = float_labels.long().to('cuda')

    optimizer.zero_grad()
    output = model(input_tensor)
    loss = F.cross_entropy(output,labels)
    loss.backward()
    optimizer.step()
    losses.append(float(loss))


# In[93]:


# get_ipython().run_line_magic('matplotlib', 'inline')


# In[94]:


plt.plot(losses)


# In[ ]:




