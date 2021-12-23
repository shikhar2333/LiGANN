#!/usr/bin/env python
import argparse
import torch
from models.networks import MultiDiscriminator, Decoder, Encoder
from models.utils import get_scheduler, weights_init, LambdaLR
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import os
import time
import molgrid
import itertools

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def return_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, train_params

def get_gmaker_eproviders(opt):
    e_pocket = molgrid.ExampleProvider(molgrid.defaultGninaReceptorTyper, data_root=opt["data_dir"],
            cache_structs=False, shuffle=True, stratify_receptor=True)
    e_ligand = molgrid.ExampleProvider(molgrid.defaultGninaLigandTyper, data_root=opt["data_dir"],
            cache_structs=False, shuffle=True, stratify_receptor=True)
    e_pocket.populate(opt["train_types_pocket"])
    e_ligand.populate(opt["train_types_ligand"])
    gmaker = molgrid.GridMaker(dimension=11.5)
    print("ligand, pocket channels: ", e_ligand.num_types(),
            e_pocket.num_types())
#     gmaker = molgrid.GridMaker()
#    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(e_ligand.num_types())
    return e_ligand, e_pocket, gmaker, dims

def write_grids(input_tensor, mol_batch, batch_no):
    center = mol_batch.coord_sets[0].center()
    src_file = mol_batch.coord_sets[0].src.replace(".gninatypes", "")
    src_file = src_file.split("/")[1]
#    print(center[0],center[1],center[2])
    for i in range(14):
        grid = molgrid.Grid3f(input_tensor[i].cpu())
        molgrid.write_dx("/scratch/shubham/grids/"+src_file+str(batch_no)+"_"+str(i)+".dx", grid, center, 0.5, 1)

def sample_grids(e_pocket_test, dims, gmaker):
    pocket_sample = e_pocket_test.next()
    pocket_tensor = torch.zeros(dims, dtype=torch.float32, device='cuda')
    gmaker.forward(pocket_sample, pocket_tensor)
    pocket_tensor = pocket_tensor.unsqueeze(0).repeat(8, 1, 1, 1, 1)
    style_ligand = torch.randn(pocket_tensor.size(0), 8, 1, 1, 1).to('cuda')
    content_pocket, _ = E1(pocket_tensor)
    pocket_to_lig = G2(content_pocket, style_ligand)
    for i in range(8):
        write_grids(pocket_to_lig[i], pocket_sample, i)

def recon_loss(target, input):
    return torch.mean(torch.abs(input - target))

if __name__ == "__main__":
    # Loss functions
    recon_loss = nn.L1Loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    recon_loss = nn.L1Loss().to(device)
    prev_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="interval between model checkpoints")

    parser.add_argument("--decay_epoch", type=int, default=1000, help="""Epoch
            from which lr decay starts""")

    parser.add_argument('--clip_gradients', type=float, default=10.0, help="Clip gradients threshold (default 10)")

    parser.add_argument("--lr_policy", default="step", help="""Policy for lr scheduling""")
    parser.add_argument("--gamma", type=float, default=0.5, help="""factor by which lr decays""")
    parser.add_argument("--step_size", type=int, default=100, help="""how often to decay learning rate""")

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
    os.makedirs(opt["save"], exist_ok=True)
    e_ligand, e_pocket, gmaker, dims = get_gmaker_eproviders(opt)

    tensor_shape = (opt["batch_size"], ) + dims
    lig_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
    pocket_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
    print("tensor shape: ", tensor_shape)

    # Initialising Encoder, Decoder and Discriminator networks
    E1, E2 = Encoder(n_downsample_s=2, n_res=2).to(device), Encoder(n_downsample_s=2, n_res=2).to(device)
    G1, G2 = Decoder(n_res=2).to(device), Decoder(n_res=2).to(device)
    total_params, train_params = return_params(E1)
    print("Total params for E1: ", total_params, "Trainable params for E1: ",
            train_params)
    total_params, train_params = return_params(G1)
    print("Total params for G1: ", total_params, "Trainable params for G1: ",
            train_params)

    params = {"n_layer": 4, "activ": "lrelu", "num_scales": 1, "pad_type":
            "replicate", "norm": "in", "bottom_dim": 32}
    D1, D2 = MultiDiscriminator(params).to(device), MultiDiscriminator(params).to(device)
    total_params, train_params = return_params(D1)
    print("Total params for D1: ", total_params, "Trainable params for D1: ",
            train_params)

    if opt["epoch"] > 0:
        E1.load_state_dict(torch.load("%s/Enc1_%d.pth" %(opt["save"],
            opt["epoch"] ) ))
        E2.load_state_dict(torch.load("%s/Enc2_%d.pth" %(opt["save"],
            opt["epoch"] ) ))
        G1.load_state_dict(torch.load("%s/Dec1_%d.pth" %(opt["save"],
            opt["epoch"] ) ))
        G2.load_state_dict(torch.load("%s/Dec2_%d.pth" %(opt["save"],
            opt["epoch"] ) ))
        D1.load_state_dict(torch.load("%s/Dis1_%d.pth" %(opt["save"],
            opt["epoch"] ) ))
        D2.load_state_dict(torch.load("%s/Dis2_%d.pth" %(opt["save"],
            opt["epoch"] ) ))
    else:
        E1.apply(weights_init())
        E2.apply(weights_init())
        G1.apply(weights_init())
        G2.apply(weights_init())
        D1.apply(weights_init())
        D2.apply(weights_init())

    # Optimizers
    optimizer_G = optim.Adam(itertools.chain(E1.parameters(), E2.parameters(),
        G1.parameters(), G2.parameters()), 
            lr=opt["lr"], betas=(opt["b1"], opt["b2"]))
    optimizer_D1 = optim.Adam(D1.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))
    optimizer_D2 = optim.Adam(D2.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))

    # LR Schedulers
    lr_scheduler_G = get_scheduler(optimizer_G, opt)
    lr_scheduler_D1 = get_scheduler(optimizer_D1, opt)
    lr_scheduler_D2 = get_scheduler(optimizer_D2, opt)

    real, fake = 1, 0
    real_D = 0.9

    # train for n_epochs
    for epoch in range(opt["epoch"], opt["n_epochs"]):
        lig_batch = e_ligand.next_batch(opt["batch_size"])
        pocket_batch = e_pocket.next_batch(opt["batch_size"])
        gmaker.forward(lig_batch, lig_tensor, 2, random_rotation=True)
        gmaker.forward(pocket_batch, pocket_tensor, 2, random_rotation=True)
#        print("input size: ", pocket_tensor.size())

        # sample style codes from unit gaussian
        style_1 = torch.randn(pocket_tensor.size(0), 8, 1, 1, 1,
                requires_grad=True).to(device)
        style_2 = torch.randn(lig_tensor.size(0), 8, 1, 1, 1,
                requires_grad=True).to(device)

        optimizer_G.zero_grad()
        c1, s1 = E1(pocket_tensor)
        c2, s2 = E2(lig_tensor)
#        print("Style shape: ", s1.size(), "Content shape: ", c1.size())

        # Reconstruct the voxels
        recon_pocket_batch = G1(c1, s1)
        recon_lig_batch = G2(c2, s2)
#        print("Reconstructed Pocket size: ", recon_pocket_batch.size(),
#                "Reconstructed Ligand size: ", recon_lig_batch.size())

        # Translate voxels
        lig_to_pocket = G1(c2, style_1)
        pocket_to_lig = G2(c1, style_2)
#        print("Translated sizes: ", lig_to_pocket.size(), pocket_to_lig.size())

        # Cyclic Translation
        c_recon_pocket, s_recon_ligand = E2(pocket_to_lig)
        c_recon_ligand, s_recon_pocket = E1(lig_to_pocket)

        cycle_recon_pocket = G1(c_recon_12, s1) if opt["lamda_cycle"] > 0 else 0
        cycle_recon_lig = G2(c_recon_21, s2) if opt["lamda_cycle"] > 0 else 0

        # Losses
        pocket_recon_loss = recon_loss(recon_pocket_batch, pocket_tensor)
        lig_recon_loss = recon_loss(recon_lig_batch, lig_tensor)*opt["lambda_reconx"]
        s1_loss = recon_loss(s_recon_21, style_1)*opt["lambda_recons"]
        s2_loss = recon_loss(s_recon_12, style_2)*opt["lambda_recons"]
        c1_loss = recon_loss(c_recon_21, c2.detach())*opt["lambda_reconc"]
        c2_loss = recon_loss(c_recon_12, c1.detach())*opt["lambda_reconc"]
        gan_loss_1 = D1.compute_loss(lig_to_pocket, real) 
        gan_loss_2 = D2.compute_loss(pocket_to_lig, real)
        cycle_loss_1 = recon_loss(cycle_recon_pocket,
                pocket_tensor)*opt["lamba_cycle"] if opt["lamda_cycle"] > 0 else 0
        cycle_loss_2 = recon_loss(cycle_recon_lig,
                lig_tensor)*opt["lamba_cycle"] if opt["lamda_cycle"] > 0 else 0

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
        nn.utils.clip_grad_norm_(list(E1.parameters()) + list(G1.parameters())
                + list(E2.parameters()) + list(G2.parameters()),
                opt["clip_gradients"])
#        for p in G1.parameters():
#            print(p.grad.norm())
        optimizer_G.step()

        # Train Discriminator 1
        optimizer_D1.zero_grad()
        loss_D1 = D1.compute_loss(pocket_tensor, real) + D1.compute_loss(lig_to_pocket.detach(), fake)
        loss_D1.backward()
        nn.utils.clip_grad_norm_(D1.parameters(), opt["clip_gradients"])
        optimizer_D1.step()

        # Train Discriminator 2
        optimizer_D2.zero_grad()
        loss_D2 = D2.compute_loss(lig_tensor, real) + D2.compute_loss(pocket_to_lig.detach(), fake)
        loss_D2.backward()
        nn.utils.clip_grad_norm_(D2.parameters(), opt["clip_gradients"])
        optimizer_D2.step()

        print(
            "[Epoch %d/%d] [Gen Loss: %f] [Dis Loss %f]" %(epoch, opt["n_epochs"], total_loss.item(), (loss_D1 + loss_D2).item() ) 
        ) 

        if lr_scheduler_G:
            lr_scheduler_G.step()
        if lr_scheduler_D1:
            lr_scheduler_D1.step()
        if lr_scheduler_D2:
            lr_scheduler_D2.step()
        
        if epoch % opt["checkpoint_interval"] == 0:
            torch.save(E1.state_dict(), "%s/Enc1_%d.pth" %(opt["save"],
                epoch))
            torch.save(E2.state_dict(), "%s/Enc2_%d.pth" %(opt["save"],
                epoch))
            torch.save(G1.state_dict(), "%s/Dec1_%d.pth" %(opt["save"],
                epoch))
            torch.save(G2.state_dict(), "%s/Dec2_%d.pth" %(opt["save"],
                epoch))
            torch.save(D1.state_dict(), "%s/Dis1_%d.pth" %(opt["save"],
                epoch))
            torch.save(D2.state_dict(), "%s/Dis2_%d.pth" %(opt["save"],
                epoch))
