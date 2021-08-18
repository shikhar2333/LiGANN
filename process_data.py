#!/usr/bin/env python
import torch
from torch.utils.data import Dataset
import molgrid
#import os
import shutil
import argparse
from rdkit import Chem, RDLogger

class CustomMolLoader(Dataset):
    def __init__(self, data_dir, voxel_transform=None, smile_transform=None):
        self.data_dir = data_dir
        self.voxel_transform = voxel_transform
        self.smile_transform = smile_transform
    
    def __len__(self) -> int:
        return 1
#        return len()

def extract_sdf_file(gninatypes_file, datadir):
    path = gninatypes_file.split("/")
    base_name = path[1].split(".")[0]
    base_name = base_name.rsplit("_", 1)[0]
    base_name += ".sdf"
    return datadir + "/structs/" + path[0] + "/" + base_name

if __name__ == "__main__":
    datadir = "/crossdock_train_data/crossdock_data"
    fname = datadir+"/training_example.types" 
    
    voxel_dir = "/crossdock_train_data/voxel_data"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=int, default=10000, help="number of iterations")
    parser.add_argument("--voxel_dir", type=str, default=voxel_dir, help="""path for
    storing voxel tensors""")
    parser.add_argument("--data_dir", type=str, default=datadir, help="""path for
    input data""")
    parser.add_argument("--types_dir", type=str, default=fname, help="""path for
    types file used for training""")
    
    opt = vars(parser.parse_args())

    molgrid.set_random_seed(0)
    torch.manual_seed(0)

    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    
    # use the libmolgrid ExampleProvider to obtain shuffled, balanced, and stratified batches from a file
    print(opt["data_dir"])
    e = molgrid.ExampleProvider(data_root=opt["data_dir"]+"/structs", cache_structs=False,shuffle=True)
    e.populate(opt["types_dir"])
    
    # initialize libmolgrid GridMaker
    gmaker = molgrid.GridMaker()
    
    # e.num_types()//2 is the number of channels used for voxel representation of docked ligand
    print("Number of channels: ", e.num_types()//2)
    dims = gmaker.grid_dimensions(e.num_types()//2)
    
    print("4D Tensor Shape: ", dims)
    tensor_shape = dims
    print(tensor_shape)
    
    # construct input tensor
    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
    
    cnt = 0
    smile_filename = "crossdock_smiles.smi"
    fp = open(smile_filename, "w")
    maxi = float('-inf')

    for iteration in range(opt["i"]):
         # load data
         batch = e.next()
         gmaker.forward(batch, input_tensor, 0, random_rotation=False)
         sdf_files = extract_sdf_file(batch.coord_sets[0].src, opt["data_dir"])
         suppl = Chem.MolFromMolFile(sdf_files)                       
         if suppl:
             tensor_filename = "voxel_tensor" + "_" + str(cnt) + ".pt"
             smile_string = Chem.MolToSmiles(suppl)
             torch.save(input_tensor, tensor_filename)
             shutil.move(tensor_filename, opt["voxel_dir"] + "/" + tensor_filename)
             maxi = max(maxi, len(smile_string))
             if smile_string != "[Y]":
                 fp.write(smile_string + "\n")
                 cnt += 1
    print(maxi)
    fp.close()
# test loading the data

#filename = "/scratch/shubham/voxel_data/" + "voxel_tensor" + "_" + "1" + ".pt"
#loaded = torch.load(filename)
#print(loaded['A'].shape, loaded['B'].shape)


#for i in range(10):
#    batch1 = e.next_batch(batch_size)
#    batch2 = e.next_batch(batch_size)
#    gmaker.forward(batch1, input_tensor1, 0, random_rotation=False)
#    gmaker.forward(batch2, input_tensor2, 0, random_rotation=False)
#    print(batch2[0].coord_sets[0].src)
#    print(batch1[0].coord_sets[0].src)
#    print()
    
