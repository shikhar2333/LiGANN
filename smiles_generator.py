import torch
from models import Shape_VAE, CNN_Encoder, MolDecoder
import molgrid
import argparse
from rdkit import Chem, RDLogger
from utils.extract_sdf_file import extract_sdf_file

vocab_list = ["pad", "start", "end",
    "C", "c", "N", "n", "S", "s", "P", "O", "o",
    "B", "F", "I","/", "\\",
    "X", "Y", "Z","W", 
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "#", "=", "-", "+", "(", ")" 
]
vocab_i2c_v1 = {i: x for i, x in enumerate(vocab_list)}
ans = set()

def extract_smiles(captions):
    smiles = set()
    print(captions)
    for caption in captions:
        smile = ''
        for i in range(1,163):
            if caption[i] == 2:
                break
            smile += vocab_i2c_v1[caption[i]]
        smiles.add(smile)
    return smiles

class GenerateLigands:
    def __init__(self, dims=(14, 48, 48, 48)) -> None:
        self.VAE = Shape_VAE(dims).to('cuda')
        self.encoder = CNN_Encoder(dims).to('cuda')
        self.decoder = MolDecoder(512, 1024, vocab_size=36).to('cuda')
        self.VAE.eval()
        self.encoder.eval()
        self.decoder.eval()
    def load_weights(self):
        self.encoder.load_state_dict(torch.load("/crossdock_train_data/saved_models/encoder_190.pth"))
        self.decoder.load_state_dict(torch.load("/crossdock_train_data/saved_models/decoder_190.pth"))
        self.VAE.load_state_dict(torch.load("/crossdock_train_data/saved_models/VAE_190.pth"))
    def return_caption(self, ligand_shapes, max_len=163):
        vectorized_rep = self.encoder(ligand_shapes)
        captions = self.decoder.beam_search_sample(vectorized_rep)
        captions = torch.stack(captions, 1)
        captions = captions.cpu().data.numpy()
        smiles = extract_smiles(captions)
        return smiles

def get_gmaker_eproviders(opt):
    e = molgrid.ExampleProvider(data_root=opt["data_dir"],
            cache_structs=False, shuffle=True)
    e.populate(opt["train_types"])
    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(e.num_types()//2)
    return e, gmaker, dims

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str,
            default="/crossdock_train_data/crossdock_data/structs",
            help="""Root dir for data""")
    parser.add_argument("--train_types", type=str,
            default="/crossdock_train_data/crossdock_data/training_example.types",
            help="train_types file path")
    opt = vars(parser.parse_args())

    input_tensor = torch.zeros((5,14,48,48,48), dtype=torch.float32, device='cuda')

    e, gmaker, dims = get_gmaker_eproviders(opt)
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    g = GenerateLigands()
    g.load_weights()
    for i in range(1):
        mol_batch = e.next_batch(5)
        gmaker.forward(mol_batch, input_tensor, 0, random_rotation=False)
        gninatype_files = [mol_batch[i].coord_sets[0].src for i in range(5)]
#        print(gninatype_files)
        sdf_files = [extract_sdf_file(gninatype_files[i], opt["data_dir"]) for i in
                range(5)]

        valid_smiles, valid_tensors = [], []
        suppl = [Chem.MolFromMolFile(sdf_files[i]) for i in range(5)] 
        for i in range(5):
            if suppl[i]:
                valid_smiles.append(Chem.MolToSmiles(suppl[i]))
                valid_tensors.append(i)
        input_tensor_modified = torch.index_select(input_tensor, 0,
                torch.tensor(valid_tensors).cuda())
        recon_x, _, _ =  g.VAE(input_tensor_modified)
        for smile in valid_smiles:
            print(smile)
        print()
        smiles = g.return_caption(recon_x)
        for smile in smiles:
            ans.add(smile)
    print(len(ans))
    for smile in ans:
        print(smile)
    
