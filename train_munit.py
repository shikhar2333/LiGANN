#!/usr/bin/env python
import argparse
import torch
from models.networks import MultiDiscriminator, Decoder, Encoder
import torch.optim as optim
import torch.nn as nn
import sys
import time
import datetime
import molgrid

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def get_gmaker_eproviders(opt):
    e_pocket = molgrid.ExampleProvider(data_root=opt["data_dir"],
            cache_structs=False, shuffle=True, stratify_receptor=True)
    e_ligand = molgrid.ExampleProvider(molgrid.defaultGninaLigandTyper, data_root=opt["data_dir"],
            cache_structs=False, shuffle=True, stratify_receptor=True)
    e_pocket.populate(opt["train_types_pocket"])
    e_ligand.populate(opt["train_types_ligand"])
#    gmaker = molgrid.GridMaker(dimension=11.5)
    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(e_ligand.num_types())
    return e_ligand, e_pocket, gmaker, dims

if __name__ == "__main__":
    # Loss functions
    recon_loss = torch.nn.L1Loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prev_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
    parser.add_argument("--lambda_recons", type=float, default=1, help="""style
            reconstruction weight""")
    parser.add_argument("--lambda_reconc", type=float, default=1, help="""content
            reconstruction weight""")
    parser.add_argument("--lambda_reconx", type=float, default=10, help="""voxel
            reconstruction weight""")
    parser.add_argument("--lamda_cycle", type=float, default=0, help="""cycle weight""")
    
    parser.add_argument("-d", "--data_dir",
    default="/scratch/shubham/v2020-other-PL", help="""Path to input
            data dir""")
    parser.add_argument("--train_types_pocket", type=str,
            default="""/scratch/shubham/training_example_pocket.types""",
            help="pocket types file for training")
    parser.add_argument("--train_types_ligand",
            default="""/scratch/shubham/training_example_ligand.types""",
            help="ligand types file for training")
    parser.add_argument("-s", "--save",
            default="/scratch/shubham/saved_models", help="""Path for saving trained models""")

    opt = vars(parser.parse_args())
    e_ligand, e_pocket, gmaker, dims = get_gmaker_eproviders(opt)

    tensor_shape = (opt["batch_size"], ) + dims
    lig_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
    pocket_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')

    # Initialising Encoder, Decoder and Discriminator networks
    E1, E2 = Encoder().to(device), Encoder().to(device)
    G1, G2 = Decoder().to(device), Decoder().to(device)

    params = {"n_layer": 4, "activ": "lrelu", "num_scales": 1, "pad_type":
            "replicate", "norm": "in", "bottom_dim": 32}
    D1, D2 = MultiDiscriminator(params).to(device), MultiDiscriminator(params).to(device)

    # Optimizers
    optimizer_G = optim.Adam(list(E1.parameters()) + list(G1.parameters()) + list(E2.parameters()) + list(G2.parameters()), 
            lr=opt["lr"], betas=(opt["b1"], opt["b2"]))
    optimizer_D1 = optim.Adam(D1.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))
    optimizer_D2 = optim.Adam(D2.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))

    real, fake = 1, 0

    # train for n_epochs
    for epoch in range(opt["n_epochs"]):
        lig_batch = e_ligand.next_batch(opt["batch_size"])
        pocket_batch = e_pocket.next_batch(opt["batch_size"])
        gmaker.forward(lig_batch, lig_tensor, 2, random_rotation=True)
        gmaker.forward(pocket_batch, pocket_tensor, 2, random_rotation=True)

        # sample style codes from unit gaussian
        style_1 = torch.randn(pocket_batch.size(0), 8, 1, 1, requires_grad=True)
        style_2 = torch.randn(lig_batch.size(0), 8, 1, 1, requires_grad=True)

        optimizer_G.zero_grad()
        c1, s1 = E1(pocket_batch)
        c2, s2 = E2(lig_batch)

        # Reconstruct the voxels
        recon_pocket_batch = G1(c1, s1)
        recon_lig_batch = G2(c2, s2)

        # Translate voxels
        lig_to_pocket = G2(c1, style_2)
        pocket_to_lig = G1(c2, style_1)

        # Cyclic Translation
        c_recon_12, s_recon_12 = E2(pocket_to_lig)
        c_recon_21, s_recon_21 = E1(lig_to_pocket)

        cycle_recon_pocket = G1(c_recon_12, s1) if opt["lamda_cycle"] > 0 else 0
        cycle_recon_lig = G2(c_recon_21, s2) if opt["lamba_cycle"] > 0 else 0

        # Losses
        pocket_recon_loss = recon_loss(recon_pocket_batch, pocket_batch)
        lig_recon_loss = recon_loss(recon_lig_batch, lig_batch)*opt["lambda_reconx"]
        s1_loss = recon_loss(s_recon_21, style_1)*opt["lambda_recons"]
        s2_loss = recon_loss(s_recon_12, style_2)*opt["lambda_recons"]
        c1_loss = recon_loss(c_recon_21, c2.detach())*opt["lambda_reconc"]
        c2_loss = recon_loss(c_recon_12, c1.detach())*opt["lambda_reconc"]
        gan_loss_1 = D1.compute_loss(lig_to_pocket, real) 
        gan_loss_2 = D2.compute_loss(pocket_to_lig, real)
        cycle_loss_1 = recon_loss(cycle_recon_pocket, pocket_batch)*opt["lamba_cycle"]
        cycle_loss_2 = recon_loss(cycle_recon_lig, lig_batch)*opt["lamba_cycle"]

        total_loss = (
                pocket_recon_loss
                + lig_recon_loss
                + s1_loss
                + s2_loss
                + c1_loss
                + c2_loss
                + gan_loss_1
                + gan_loss_2
                + cycle_loss_1
                + cycle_loss_2
        )
        total_loss.backward()
        optimizer_G.step()

        # Train Discriminator 1
        optimizer_D1.zero_grad()
        loss_D1 = D1.compute_loss(pocket_batch, real) + D1.compute_loss(lig_to_pocket.detach(), fake)
        loss_D1.backward()
        optimizer_D1.step()

        # Train Discriminator 2
        optimizer_D2.zero_grad()
        loss_D2 = D2.compute_loss(lig_batch, real) + D2.compute_loss(pocket_to_lig.detach(), fake)
        loss_D2.backward()
        optimizer_D2.step()

        print(
            "[Epoch %d/%d] [Gen Loss: %f] [Dis Loss %f]" %(epoch, opt["n_epochs"], total_loss.item(), (loss_D1 + loss_D2).item() ) 
        ) 
