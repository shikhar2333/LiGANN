from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import os


def generate_representation(in_smile, filename):
    """
    Makes embeddings of Molecule.
    """
    try:
        m = Chem.MolFromSmiles(in_smile)
        mh = Chem.AddHs(m)
        AllChem.EmbedMolecule(mh)
        Chem.AllChem.MMFFOptimizeMolecule(mh)
        m = Chem.RemoveHs(mh)
        path = os.path.join("/scratch/shubham/zinc", str(filename))
        os.makedirs(path)
        with open(path+"/"+str(filename)+".txt",'w+') as f:
            f.write(in_smile)

        writer = Chem.SDWriter("/scratch/shubham/zinc/"+str(filename)+"/"+str(filename)+".sdf")
        writer.write(m)
        writer.close()
    except:
        with open("/scratch/shubham/invalid.txt",'w+') as f:
            f.write(str(filename) + " " + in_smile)



file = open("/scratch/shubham/smiles.smi",'r')

cnt=1
for smile in file:
    print(cnt, flush=True)
    generate_representation(smile,cnt)
    cnt+=1
