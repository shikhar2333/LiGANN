#!/usr/bin/env python
import molgrid
import argparse
from rdkit import Chem, RDLogger
import selfies as sf

def extract_sdf_file(gninatypes_file, datadir):
    path = gninatypes_file.split("/")
    base_name = path[1].split(".")[0]
    base_name = base_name.rsplit("_", 1)[0]
    base_name += ".sdf"
    return datadir + "/structs/" + path[0] + "/" + base_name

if __name__ == "__main__":
    datadir = "/scratch/shubham/crossdock_data"
    fname = datadir+"/training_example.types" 

    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=int, default=10000, help="number of iterations")
    parser.add_argument("--data_dir", type=str, default=datadir, help="""path for
    input data""")
    parser.add_argument("--train_types", type=str, default=fname, help="""path for
    types file used for training""")
    
    opt = vars(parser.parse_args())

    molgrid.set_random_seed(0)
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    cnt = 0
    maxi = float('-inf')
    fp = open("training_example.types", "w")
    selfies_set = set()
    with open(opt["train_types"], "r") as types_file:
        for line in types_file:
            # extract the ligand file from the second last col
            docked_ligand_file = line.split()[-2]
            # get the corresponding sdf file
            sdf_file = extract_sdf_file(docked_ligand_file, opt["data_dir"])
            mol_object = Chem.MolFromMolFile(sdf_file)
            if mol_object:
                smile_str = Chem.MolToSmiles(mol_object)
                selfies_str = sf.encoder(smile_str)
                if selfies_str:
                    line = line[:-1]
                    fp.write(line + "\n")
                    selfies_set.add(selfies_str)
    print("DONE", len(selfies_set))
    selfies_file = open("/scratch/shubham/crossdock_data/selfies.txt", "w")
    for selfies in selfies_set:
        selfies_file.write(selfies+"\n")
    fp.close()
    selfies_file.close()
