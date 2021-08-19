import argparse
import os
import molgrid
from models import Shape_VAE, CNN_Encoder, MolDecoder 
import torch
import torch.optim as optim
import torch.nn as nn
from utils.process_smiles import process_smiles
from utils.extract_sdf_file import extract_sdf_file
from rdkit import Chem

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def get_gmaker_eproviders(opt):
    e = molgrid.ExampleProvider(data_root=opt["data_dir"],
            cache_structs=False, shuffle=True)
    e.populate(opt["train_types"])
    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(e.num_types()//2)
    return e, gmaker, dims

if __name__ == "__main__":
    # Loss functions
    mae_loss = torch.nn.L1Loss()
    cuda = True if torch.cuda.is_available() else False

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("-d", "--data_dir", type=str,
            default="/crossdock_train_data/crossdock_data/structs",
            help="""Root dir for data""")
    parser.add_argument("--train_types", type=str,
            default="/crossdock_train_data/crossdock_data/training_example.types",
            help="train_types file path")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
    parser.add_argument("--lambda_pixel", type=float, default=10, help="pixelwise loss weight")
    parser.add_argument("--lambda_latent", type=float, default=0.1, help="latent loss weight")
    parser.add_argument("--lambda_kl", type=float, default=0.01, help="kullback-leibler loss weight")
    
    parser.add_argument("-v", "--voxels",
    default="/crossdock_train_data/voxel_data", help="""Path to input
            voxel data""")
    parser.add_argument("-s", "--save",
            default="/crossdock_train_data/saved_models", help="""Path for saving trained models""")
    opt = vars(parser.parse_args())
    
    os.makedirs(opt["save"], exist_ok=True)
    
    e, gmaker, dims = get_gmaker_eproviders(opt)

    VAE = Shape_VAE(dims).to('cuda')
    encoder = CNN_Encoder(dims).to('cuda')

    if opt["epoch"] > 0:
        # load saved models
        VAE.load_state_dict(torch.load("%s/VAE_%d.pth" %(opt["save"], 
            opt["epoch"]) ))
        encoder.load_state_dict(torch.load("%s/encoder_%d.pth" %(opt["save"], 
            opt["epoch"]) ))
    else:
        # Initialise weights
        VAE.apply(init_weights)
        encoder.apply(init_weights)

    # construct optimizers for the networks
    optimizer_VAE = optim.Adam(VAE.parameters(), lr=opt["lr"], betas=(opt["b1"],
        opt["b2"]))

    tensor_shape = (opt["batch_size"], ) + dims
    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')

    # train for n_epochs
    for epoch in range(opt["n_epochs"]):
        mol_batch = e.next_batch(opt["batch_size"])
        gmaker.forward(mol_batch, input_tensor, 0, random_rotation=False)

        gninatype_files = [mol_batch[i].coord_sets[0].src for i in range(opt["batch_size"])]
        sdf_files = [extract_sdf_file(gninatype_files[i], opt["data_dir"]) for i in
                range(opt["batch_size"])]

        valid_smiles, valid_tensors = [], []
        suppl = [Chem.MolFromMolFile(sdf_files[i]) for i in range(opt["batch_size"])] 
        for i in range(opt["batch_size"]):
            if suppl[i]:
                valid_smiles.append(Chem.MolToSmiles(suppl[i]))
                valid_tensors.append(i)

        valid_smiles, lengths = process_smiles(valid_smiles)
        input_tensor_modified = torch.index_select(input_tensor, 0,
                torch.tensor(valid_tensors).cuda())

        optimizer_VAE.zero_grad()
        recon_x, mu, logvar = VAE(input_tensor_modified)
        total_loss, BCE_loss, KLD = VAE.loss(recon_x,
                    input_tensor_modified, mu, logvar)
        total_loss.backward()
        optimizer_VAE.step()
        print("[Epoch %d/%d] [VAE Loss: %f]" %(epoch,
                    opt["n_epochs"], total_loss.item()) )
        features = encoder(recon_x)
        print(features.shape)
    
        if epoch % opt["checkpoint_interval"] == 0:
            torch.save(VAE.state_dict(), "%s/VAE_%d.pth"
                    %(opt["save"], epoch))
            torch.save(encoder.state_dict(), "%s/encoder_%d.pth"
                    %(opt["save"], epoch))

