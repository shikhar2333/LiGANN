import argparse
import sys
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import numpy as np
from models_shape import CNN_Encoder, MolDecoder
import molgrid
import matplotlib.pyplot as plt
import torch.optim as optim

from utils.process_selfies import encode_selfies
from utils.extract_sdf_file import extract_sdf_file
import selfies as sf
from rdkit import Chem, RDLogger


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def get_gmaker_eproviders(opt):
    e = molgrid.ExampleProvider(molgrid.defaultGninaLigandTyper,data_root=opt["data_dir"],
            cache_structs=False, shuffle=True, stratify_receptor=True,)
    e.populate(opt["train_types"])
    gmaker = molgrid.GridMaker(dimension=11.5)
    dims = gmaker.grid_dimensions(e.num_types())
    return e, gmaker, dims

def loss_function(recon_x, x, mu, logvar):
    reconstruction_function = nn.BCELoss()
    reconstruction_function.size_average = False
    BCE = reconstruction_function(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD

def plot_loss(iterations, loss_values):
    plt.plot(iterations, loss_values)
    plt.show()

def generate_representation(sdf_files, input_tensor, batch_size):
    suppl = [Chem.MolFromMolFile(sdf_files[i]) for i in range(batch_size)]
    selfies = []
    for i in range(batch_size):
        smile_str = Chem.MolToSmiles(suppl[i])
        selfies.append(sf.encoder(smile_str)) 
    
    input_tensor, captions, lengths = encode_selfies(selfies, input_tensor)
    return input_tensor, captions, lengths

if __name__ == "__main__":
    # Loss function
    cross_entropy_loss = nn.CrossEntropyLoss()
    loss, iters = [], []
    cuda = True if torch.cuda.is_available() else False
    caption_start = 0
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--cap_start", type=int, default=caption_start,
            help="epoch at which caption training starts")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("-d", "--data_dir", type=str,
            default="/scratch/shubham/zinc",
            help="""Root dir for data""")
    parser.add_argument("--train_types", type=str,
            default="/scratch/shubham/zinc_training_example.types",
            help="train_types file path")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="interval between model checkpoints")
    
    parser.add_argument("-s", "--save",
            default="/scratch/shubham/saved_models", help="""Path for saving trained models""")
    opt = vars(parser.parse_args())
    
    os.makedirs(opt["save"], exist_ok=True)
    
    e, gmaker, dims = get_gmaker_eproviders(opt)

    encoder = CNN_Encoder(14)
    decoder = MolDecoder(512, 512, 512)
    encoder.cuda()
    decoder.cuda()

    criterion = nn.CrossEntropyLoss().to('cuda')
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
    
    encoder.train()
    decoder.train()


    tensor_shape = (opt["batch_size"], ) + dims
    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
    caption_loss = 0.
    torch.autograd.set_detect_anomaly(True)
    loss_file = open("/scratch/shubham/loss.txt", "w")

    # train for n_epochs
    for epoch in range(opt["epoch"], opt["epoch"]+opt["n_epochs"]):
        mol_batch = e.next_batch(opt["batch_size"])
        gmaker.forward(mol_batch, input_tensor, 2, random_rotation=True)
        
        
        gninatype_files = [mol_batch[i].coord_sets[0].src for i in range(opt["batch_size"])]
        sdf_files = [extract_sdf_file(gninatype_files[i], opt["data_dir"]) for i in range(opt["batch_size"])]
        input_tensor, captions, lengths = generate_representation(sdf_files, input_tensor, opt["batch_size"])
        in_data = Variable(input_tensor[:, :14])
        captions = captions.cuda()
        in_data = in_data.cuda()
        lengths = [length - 1 for length in lengths]

        features = encoder(in_data)
        outputs, alphas, decode_lengths = decoder(features, captions, lengths)
        targets = pack_padded_sequence(captions[:, 1:],decode_lengths, batch_first=True)[0]
        outputs = pack_padded_sequence(outputs ,decode_lengths, batch_first=True)[0]
        cap_loss = criterion(outputs, targets)
        cap_loss += ((1. - alphas.sum(dim=1)) ** 2).mean()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        cap_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        

        if (epoch + 1)%10 == 0:
            print("[Epoch %d/%d] [Caption Loss: %f]" %(epoch + 1,
                        opt["epoch"] + opt["n_epochs"], cap_loss))
            sys.stdout.flush()
            res = "[Epoch %d/%d] [Caption Loss: %f]" %(epoch + 1,
                        opt["epoch"] + opt["n_epochs"], cap_loss)
            loss_file.write(res+'\n')
            loss_file.flush()
    
        if epoch % opt["checkpoint_interval"] == 0:
            if epoch >= opt["cap_start"]:
                torch.save(encoder.state_dict(), "%s/encoder_%d.pth"
                        %(opt["save"], epoch))
                torch.save(decoder.state_dict(), "%s/decoder_%d.pth"
                    %(opt["save"], epoch))

        
    loss_file.close()


