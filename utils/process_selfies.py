import numpy as np
import selfies as sf
import torch

global_alphabet = ['pad', 'start', 'end', '[#C]', '[#N]', '[/B]', '[/Br]', '[/C@@Hexpl]', '[/C@@expl]', '[/C@Hexpl]', '[/C@expl]', '[/C]', '[/Cl]', '[/F]', '[/N]', '[/O]', '[/P@expl]', '[/P]', '[/S@@Hexpl]', '[/SHexpl]', '[/S]', '[=C]', '[=Moexpl]', '[=N+expl]', '[=N]', '[=O]', '[=P@@expl]', '[=P@expl]', '[=P]', '[=S@expl]', '[=SHexpl]', '[=S]', '[=Vexpl]', '[AlH2expl]', '[Asexpl]', '[Auexpl]', '[B]', '[Br]', '[Branch1_1]', '[Branch1_2]', '[Branch1_3]', '[Branch2_1]', '[Branch2_2]', '[Branch2_3]', '[C@@Hexpl]', '[C@@expl]', '[C@Hexpl]', '[C@expl]', '[C]', '[Cl]', '[Crexpl]', '[Expl/Ring2]', '[Expl=Ring1]', '[Expl=Ring2]', '[Expl\\Ring1]', '[Expl\\Ring2]', '[F]', '[Fe@@expl]', '[Fe@expl]', '[Feexpl]', '[Hgexpl]', '[I]', '[Liexpl]', '[Mgexpl]', '[Moexpl]', '[N+expl]', '[N@@H+expl]', '[NH+expl]', '[NHexpl]', '[N]', '[O-expl]', '[O]', '[P@@expl]', '[P@expl]', '[PHexpl]', '[P]', '[Ring1]', '[Ring2]', '[Ru@@expl]', '[Ruexpl]', '[S@@Hexpl]', '[S@@expl]', '[S@Hexpl]', '[S@expl]', '[SHexpl]', '[S]', '[Sbexpl]', '[Scexpl]', '[SeHexpl]', '[Seexpl]', '[Siexpl]', '[SnHexpl]', '[Vexpl]', '[Wexpl]', '[Znexpl]', '[\\B]', '[\\C@@Hexpl]', '[\\C@@expl]', '[\\C@Hexpl]', '[\\C@expl]', '[\\C]', '[\\F]', '[\\N+expl]', '[\\N]', '[\\O]', '[\\P]', '[\\S]']

global_pad_to_len = 183
global_symbol_to_idx = {s: i for i, s in enumerate(global_alphabet)}

def encode_selfies(selfies, input_tensor):
    lenghts = [sf.len_selfies(s) + 2 for s in selfies]
    # print(lenghts)
    lenghts = np.array(lenghts)
    indices = np.argsort(lenghts)[::-1]
    lenghts = lenghts[indices]
    encoded_selfies = np.zeros(( len(selfies), global_pad_to_len))
    sorted_selfies = [selfies[idx] for idx in indices]
    for i, selfie in enumerate(sorted_selfies):
        symbols = sf.split_selfies(selfie)
        selfie_temp = [1] + [global_symbol_to_idx[symbol] for symbol in symbols] + [2]
        encoded_selfies[i, :len(selfie_temp)] = selfie_temp
    return input_tensor[list(indices)], torch.tensor(encoded_selfies).long(), lenghts 

if __name__ == "__main__":
    dataset = []
    with open("selfies.txt") as fp:
        for line in fp:
            line = line.strip()
            dataset.append(line)
    alphabet = sf.get_alphabet_from_selfies(dataset)
    alphabet = list(sorted(alphabet))
    alphabet = ["pad", "start", "end"] + alphabet
    pad_to_len = max(sf.len_selfies(s) + 2 for s in dataset)  
    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
    selfies = encode_selfies(dataset[0:2])
    print(selfies)
