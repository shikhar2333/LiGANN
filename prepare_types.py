import argparse
import os
import random
parser = argparse.ArgumentParser()

# parser.add_argument("--types", type=str,
#         default="/scratch/shubham/crossdock_data/it2_tt_completeset_train0.types", help="""path for the
#         types file""")

# parser.add_argument("--output", type=str, default="types", help="path for the output directory")

# opt = parser.parse_args()

# os.makedirs(opt.output, exist_ok=True)

dir = "/scratch/shubham/zinc"
with open("/scratch/shubham/zinc_training_example.types","a") as fp:
    for ligand in os.listdir(dir):
        print(ligand)
        fp.write(ligand+"/"+ligand+".gninatypes"+"\n")


# with open(opt.output+"/training_example.types","w") as fp:
#     with open(opt.types) as fp_types:
#         for line in fp_types:
#             first_col = line.split()[0]
#             if first_col == "1":
#                 fp.write(line)
