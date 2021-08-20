# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license
import numpy as np
from rdkit import Chem
import torch
def process_smiles(smiles):
    lengths = [len(smile) for smile in smiles]
    strings = np.zeros(( len(smiles), max(lengths) + 2 ))
    lengths.sort(reverse=True)
    lengths = np.array(lengths)

    vocab_list = ["pad", "start", "end",
        "C", "c", "N", "n", "S", "s", "P", "O", "o",
        "B", "F", "I","/", "\\",
        "X", "Y", "Z","W", "H",
        "1", "2", "3", "4", "5", "6", "7", "8",
        "#", "=", "-", "+", "(", ")", "[", "]", "@"
    ]
    vocab_i2c_v1 = {i: x for i, x in enumerate(vocab_list)}
    vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}
    
    for i, sstring in enumerate(smiles):
        mol = Chem.MolFromSmiles(sstring)
        if not mol:
            raise ValueError("Failed to parse molecule '{}'".format(mol))
    
        sstring = Chem.MolToSmiles(mol)  # Make the SMILES canonical.
        sstring = sstring.replace("Cl", "X").replace("[nH]", "Y").replace("Br",
                "Z").replace("Ru", "W")
        try:
            vals = [1] + [vocab_c2i_v1[xchar] for xchar in sstring] + [2]
        except KeyError:
            raise ValueError(("Unkown SMILES tokens: {} in string '{}'."
                              .format(", ".join([x for x in sstring if x not in vocab_c2i_v1]),
                                                                          sstring)))
        strings[i, :len(vals)] = vals
    
    return torch.tensor(strings).long(), lengths
