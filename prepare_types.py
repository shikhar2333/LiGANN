import argparse
import os
import random
parser = argparse.ArgumentParser()

parser.add_argument("--structs", type=str,
        default="/scratch/shubham/crossdock_data/structs", help="path for the structs dir")

parser.add_argument("--output", type=str, default="types", help="path for the output directory")

parser.add_argument("--n", type=int, default=26, help="number of protein_dirs to use")

opt = parser.parse_args()

if not os.path.isdir(opt.output):
    os.makedirs(opt.output)

protein_dirs = []
for dir in os.listdir(opt.structs):
     protein_dirs.append(dir)

random.shuffle(protein_dirs)
protein_dirs = protein_dirs[:opt.n]
print(opt.output)

with open(opt.output+"/training_example.types","w") as fp:
    for dir in protein_dirs:
        for file in os.listdir(opt.structs+"/"+dir):
            if "gninatypes" in file:
                fp.write("0 " + dir + "/" + file+'\n')


