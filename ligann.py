#!/usr/bin/env python
import argparse
from models import *
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import sys
import time
import datetime
from MolDataLoader import CustomMolLoader
from utils.process_smiles import process_smiles

Tensor = torch.cuda.FloatTensor
def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt["latent_dim"]))))
    z = sampled_z * std + mu
    return z

#total_params = sum(p.numel() for p in generator.parameters())
#train_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
#print("Total Parameters: ", total_params)
#print("Trainable Parameters: ", train_params)

#total_params = sum(p.numel() for p in D_VAE.parameters())
#train_params = sum(p.numel() for p in D_VAE.parameters() if p.requires_grad)
#print("Total Parameters: ", total_params)
#print("Trainable Parameters: ", train_params)


if __name__ == "__main__":
    # Loss functions
    mae_loss = torch.nn.L1Loss()
    cuda = True if torch.cuda.is_available() else False
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
    parser.add_argument("--lambda_pixel", type=float, default=10, help="pixelwise loss weight")
    parser.add_argument("--lambda_latent", type=float, default=0.1, help="latent loss weight")
    parser.add_argument("--lambda_kl", type=float, default=0.01, help="kullback-leibler loss weight")
    
    parser.add_argument("-i", "--input", required=True, help="Path to input .smi file.")
    parser.add_argument("-v", "--voxels",
    default="/crossdock_train_data/voxel_data", help="""Path to input
            voxel data""")
    parser.add_argument("-s", "--save",
            default="/crossdock_train_data/saved_models", help="""Path for saving trained models""")
    opt = vars(parser.parse_args())
    
    smiles_path = opt["input"]
    dims = (14, 48, 48, 48)

    strings, lengths = process_smiles(smiles_path)
    molDatasetObject = CustomMolLoader(opt["voxels"], strings, lengths)

    # initialize CustomMolLoader
    dataloader = DataLoader(molDatasetObject, batch_size=opt["batch_size"], shuffle=True)
    print(len(dataloader))

    # Initialize Generator, Enocoder, VAE and LR Discriminator on GPU
    generator = Generator(8, dims).to('cuda')
    encoder = Encoder(vaeLike=True).to('cuda')
    D_VAE = MultiDiscriminator(dims).to('cuda')
    D_LR = MultiDiscriminator(dims).to('cuda')
    VAE = Shape_VAE(dims).to('cuda')

    if opt["epoch"] > 0:
        # load saved models
        generator.load_state_dict(torch.load("%s/generator_%d.pth" %(opt["save"],
            opt["epoch"]) ))
        encoder.load_state_dict(torch.load("%s/encoder_%d.pth" %(opt["save"],
            opt["epoch"] )))
        D_VAE.load_state_dict(torch.load("%s/D_VAE_%d.pth" %(opt["save"],
            opt["epoch"] )))
        D_LR.load_state_dict(torch.load("%s/D_LR_%d.pth" %(opt["save"],
            opt["epoch"] )))
        VAE.load_state_dict(torch.load("%s/VAE_%d.pth" %(opt["save"], 
            opt["epoch"]) ))
    else:
        # Initialise weights
        generator.apply(weights_init)
        D_VAE.apply(weights_init)
        D_LR.apply(weights_init)
        VAE.apply(weights_init)

    # construct optimizers for the 4 networks
    optimizer_G = optim.Adam(generator.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))
    optimizer_E = optim.Adam(encoder.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))
    optimizer_D_VAE = optim.Adam(D_VAE.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))
    optimizer_D_LR = optim.Adam(D_LR.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))
    optimizer_VAE = optim.Adam(VAE.parameters(), lr=opt["lr"], betas=(opt["b1"],
        opt["b2"]))

    # train for n_epochs
    for epoch in range(opt["n_epochs"]):
        for i, batch in enumerate(dataloader):
            input_tensor1 = batch["A"][0]
            input_tensor2 = batch["B"][0]            # Training the encoder and generator
            optimizer_E.zero_grad()
            optimizer_G.zero_grad()
            
            mu, logvar = encoder(input_tensor2)
            encoded_z = reparameterization(mu, logvar)
    
            fake_ligands = generator(input_tensor1, encoded_z)
            
            # L1 loss for measuring degree of diff between generated outputs and the actual input
            loss_pixel = mae_loss(fake_ligands, input_tensor2)
            
            # KL divergence between the distribution learned by the encoder and a random Gaussian
            loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
            
            # discrimantor distinguishing b/w fake and real for cVAE GAN
            loss_VAE_GAN = D_VAE.compute_loss(fake_ligands, 1)
            
            # sample z values from the Gaussian distribution with mu,sigma = 0, 1
            sampled_z = Variable(Tensor(np.random.normal(0, 1, (input_tensor1.size(0), opt["latent_dim"]))))
            fake_ligands1 = generator(input_tensor1, sampled_z)
            
            # discrimantor distinguishing b/w fake and real for cLR GAN
            loss_LR_GAN = D_LR.compute_loss(fake_ligands1, 1)
            
            # Total Loss: Generator + Encoder
            loss_GE = loss_VAE_GAN + loss_LR_GAN + opt['lambda_pixel'] * loss_pixel + opt['lambda_kl'] * loss_kl
            
            loss_GE.backward(retain_graph=True)
            optimizer_E.step()
            
            # Generator only loss
            _mu, _ = encoder(fake_ligands1)
            loss_latent = opt['lambda_latent'] * mae_loss(_mu, sampled_z)
            
            loss_latent.backward()
            optimizer_G.step()
    
            # Feed the generator output to the Shape_VAE network
            optimizer_VAE.zero_grad()
            recon_x, mu, logvar = VAE(fake_ligands.detach())
            total_loss, BCE_loss, KLD = VAE.loss(recon_x,
                    fake_ligands.detach(), mu, logvar)
            total_loss.backward()
            optimizer_VAE.step()
    
            # Train the discriminator for the cVAE GAN.
            optimizer_D_VAE.zero_grad()
            loss_D_VAE = D_VAE.compute_loss(fake_ligands.detach(), 0) + D_VAE.compute_loss(input_tensor2, 1)
            
            loss_D_VAE.backward()
            optimizer_D_VAE.step()
            
            # Train the discriminator for the cLR GAN.
            optimizer_D_LR.zero_grad()
            loss_D_LR = D_LR.compute_loss(fake_ligands1.detach(), 0) + D_LR.compute_loss(input_tensor2, 1)
            
            loss_D_LR.backward()
            optimizer_D_LR.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt["n_epochs"]* len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
 
            sys.stdout.write(
                    """\r[Epoch %d/%d] [Batch %d/%d] [G loss: %.3f, pixel loss: %.3f, kl loss: %.3f,
                    latent loss: %.3f, D_VAE loss: %.3f, D_LR loss: %.3f, Shape_VAE loss:
                    %.3f] ETA: %s"""
                    % (
                        epoch,
                        opt["n_epochs"],
                        i,
                        len(dataloader),
                        loss_GE.item(),
                        loss_pixel.item(),
                        loss_kl.item(),
                        loss_latent.item(),
                        loss_D_VAE.item(),
                        loss_D_LR.item(),
                        total_loss.item(),
                        time_left,
                    )
            )
            if i == 100:
                break
        if epoch % opt["checkpoint_interval"] == 0:
            torch.save(generator.state_dict(), "%s/generator_%d.pth"
                    %(opt["save"], epoch))
            torch.save(encoder.state_dict(), "%s/encoder_%d.pth"
                    %(opt["save"], epoch))
            torch.save(D_VAE.state_dict(), "%s/D_VAE_%d.pth"
                    %(opt["save"], epoch))
            torch.save(D_LR.state_dict(), "%s/D_LR_%d.pth"
                    %(opt["save"], epoch))
            torch.save(VAE.state_dict(), "%s/VAE_%d.pth"
                    %(opt["save"], epoch))

