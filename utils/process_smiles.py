# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license
import numpy as np
from rdkit import Chem
import torch
vocab_list = ["pad", "start", "end",
    "C", "c", "N", "n", "S", "s", "P", "O", "o",
    "B", "F", "I","/", "\\",
    "X", "Y", "Z","W", 
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "#", "=", "-", "+", "(", ")"
]
vocab_i2c_v1 = {i: x for i, x in enumerate(vocab_list)}
vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}
#for i in range(39):
#    print(i,vocab_i2c_v1[i])
def process_smiles(smiles):
#    for smile in smiles:
#        print(smile, len(smile))
    mod_smiles = []
#    print()
    for smile in smiles:
        sstring = smile.replace("[", "").replace("]", "").replace("@", "").replace("@@", "").replace("H","")
        sstring = sstring.replace("Cl", "X").replace("[nH]", "Y").replace("Br",
                "Z").replace("Ru", "W")
#        print(sstring, len(sstring))
        mod_smiles.append(sstring)
    lengths = [len(smile)+2 for smile in mod_smiles]
    strings = np.zeros(( len(smiles), max(lengths) ))
    lengths = np.array(lengths)
    indices = np.argsort(lengths)[::-1]
    lengths = lengths[indices]

    temp = []
    for idx in indices:
        temp.append(smiles[idx])
#        print(mod_smiles[idx], len(mod_smiles[idx]))
#        print(len(smiles[idx]))
#    temp = temp[::-1]
#    temp_len = [len(smile) for smile in temp]
#    print(temp_len)
    for i, sstring in enumerate(temp):
        mol = Chem.MolFromSmiles(sstring)
        if not mol:
            raise ValueError("Failed to parse molecule '{}'".format(mol))
    
        sstring = Chem.MolToSmiles(mol)  # Make the SMILES canonical.
        sstring = sstring.replace("[", "").replace("]", "").replace("@", "").replace("@@", "").replace("H",
                "")
        sstring = sstring.replace("Cl", "X").replace("[nH]", "Y").replace("Br",
                "Z").replace("Ru", "W")
#        print(len(sstring))
        try:
            vals = [1] + [vocab_c2i_v1[xchar] for xchar in sstring] + [2]
        except KeyError:
            raise ValueError(("Unkown SMILES tokens: {} in string '{}'."
                              .format(", ".join([x for x in sstring if x not in vocab_c2i_v1]),
                                                                          sstring)))
        strings[i, :len(vals)] = vals

    return torch.tensor(strings).long(), lengths
