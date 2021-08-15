#!/usr/bin/env python
import torch
import molgrid
import os
import shutil

batch_size = 5
#datadir = '/scratch/shubham/crossdock_data'
datadir = "/crossdock_train_data/crossdock_data"
fname = datadir+"/training_example.types" 

molgrid.set_random_seed(0)
torch.manual_seed(0)
# np.random.seed(0)

# use the libmolgrid ExampleProvider to obtain shuffled, balanced, and stratified batches from a file
e = molgrid.ExampleProvider(data_root=datadir+"/structs", cache_structs=False,shuffle=True)
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

#filename = "/scratch/shubham/voxel_data/" + "voxel_tensor" + "_" + "1" + ".pt"
#loaded = torch.load(filename)
#print(loaded['A'].shape, loaded['B'].shape)

def extract_sdf_file(gninatypes_file):
    path = gninatypes_file.split("/")
    base_name = path[1].split(".")[0]
    base_name = base_name.rsplit("_", 1)[0]
    base_name += ".sdf"
    return datadir + "/structs/" + path[0] + "/" + base_name

for i in range(10):
    batch1 = e.next_batch(batch_size)
    batch2 = e.next_batch(batch_size)
    gmaker.forward(batch1, input_tensor1, 0, random_rotation=False)
    gmaker.forward(batch2, input_tensor2, 0, random_rotation=False)
    print(batch2[0].coord_sets[0].src)
    print(batch1[0].coord_sets[0].src)
    print()
    
