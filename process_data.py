#!/usr/bin/env python
# coding: utf-8
import torch
import molgrid
import os
import shutil

batch_size = 5
datadir = '/scratch/shubham/crossdock_data'
fname = datadir+"/custom_cd.types" 

molgrid.set_random_seed(0)
torch.manual_seed(0)
# np.random.seed(0)

# use the libmolgrid ExampleProvider to obtain shuffled, balanced, and stratified batches from a file
e = molgrid.ExampleProvider(data_root=datadir+"/structs", cache_structs=False, balanced=True,shuffle=True)
e.populate(fname)

# initialize libmolgrid GridMaker
gmaker = molgrid.GridMaker()

# e.num_types()//2 is the number of channels used for voxel representation of docked ligand
print("Number of channels: ", e.num_types()//2)
dims = gmaker.grid_dimensions(e.num_types()//2)

print("4D Tensor Shape: ", dims)
tensor_shape = (batch_size,)+dims
print(tensor_shape)

# construct input tensors
input_tensor1 = torch.zeros(tensor_shape, dtype=torch.float32)
input_tensor2 = torch.zeros(tensor_shape, dtype=torch.float32)

'''
    Generate voxel data batchwise( 5 batches default ) for 500 epochs
'''

# for iteration in range(500):
#     # load data
#     batch1 = e.next_batch(batch_size)
#     batch2 = e.next_batch(batch_size)
#     # libmolgrid can interoperate directly with Torch tensors, using views over the same memory.
#     # internally, the libmolgrid GridMaker can use libmolgrid Transforms to apply random rotations and translations for data augmentation
#     # the user may also use libmolgrid Transforms directly in python
#     gmaker.forward(batch1, input_tensor1, 0, random_rotation=False)
#     gmaker.forward(batch2, input_tensor2, 0, random_rotation=False)
#     torch_dict = {'A': input_tensor1, 'B': input_tensor2}
#     filename = "voxel_tensor" + "_" + str(iteration) + ".pt"
#     torch.save(torch_dict, filename)
#     shutil.move(filename, "/scratch/shubham/voxel_data/" + filename)

# test loading the data
filename = "/scratch/shubham/voxel_data/" + "voxel_tensor" + "_" + "1" + ".pt"
loaded = torch.load(filename)
print(loaded['A'].shape, loaded['B'].shape)


