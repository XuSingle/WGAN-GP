from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from model import Generator, Discriminator
from utils import *
from pytorch_gan_metrics import get_inception_score
# from IvO import *

def main():
    # Create the dataset
    dataset = dset.CIFAR10(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(args.image_size),
                                   transforms.CenterCrop(args.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]), download=True, train=True)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.workers)

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    vutils.save_image(real_batch[0][:64], './result/real_sample.png', padding=5, normalize=True)

    # Create the generator
    netG = Generator(1, args.noise_size, args.ngf, 3).to(device)
    # Create the Discriminator
    netD = Discriminator(1, args.ndf, 3).to(device)

    if not args.load:
        netG.apply(weights_init)
        netD.apply(weights_init)
    else:
        netG.load_state_dict(torch.load('models/netG_WGAN_GP_epoch_%d.pth' % args.load_epoch))
        netD.load_state_dict(torch.load('models/netD_WGAN_GP_epoch_%d.pth' % args.load_epoch))

    fixed_noise = torch.randn(64, args.noise_size, 1, 1, device=device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    img_list = []
    fid_record = []
    IS_record = []
    IS_std_record = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(args.epoch):
        # For each batch in the dataloader
        netD.train()
        netG.train()
        for i, data in enumerate(dataloader, 0):

            optimizerD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            # Forward pass real batch through D
            real_validity = netD(real_cpu).view(-1)

            noise = torch.randn(b_size, args.noise_size, 1, 1, device=device)
            fake = netG(noise)

            fake_validity = netD(fake.detach()).view(-1)

            gradient_penalty = args.lambda_gp * compute_gradient_penalty(netD, real_cpu, fake)
            # Adversarial loss
            errD = - torch.mean(real_validity) + torch.mean(fake_validity)
            errD_gp = errD + gradient_penalty

            errD_gp.backward()

            optimizerD.step()

            if i % args.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                optimizerG.zero_grad()
                fake_imgs = netG(noise)
                fake_validity = netD(fake_imgs)
                errG = -torch.mean(fake_validity)

                errG.backward()
                optimizerG.step()


            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tGP: %.4f' % (epoch, args.epoch, i, len(dataloader),
                         errD.item(), errG.item(), gradient_penalty.item()))

        if epoch > 1 and (epoch % 5 == 0 or epoch == args.epoch - 1):
            if args.epoch - 1 == epoch:
                cur_epoch = epoch + 1 + args.load_epoch
            else:
                cur_epoch = epoch + args.load_epoch
            netD.eval()
            netG.eval()
            with torch.no_grad():

                torch.cuda.empty_cache()
                fake = netG(fixed_noise).detach().cpu()
                vutils.save_image(vutils.make_grid(fake, padding=2, normalize=True),
                                  './result/epoch%d_result.png' % cur_epoch)

                # Because of the limitation of CUDA memory, 5 batch to generate 10,000 fake pictures.

                eva_size = 64
                Noisee = torch.randn(eva_size, args.noise_size, 1, 1, device=device)
                fake = netG(Noisee)
                for i in range(eva_size):
                    vutils.save_image(fake[i].detach(), './evaluation/random_%s.png' % (str(i)), normalize=True)

                for ii in range(10000 // 64):
                    Noisee = torch.randn(eva_size, args.noise_size, 1, 1, device=device)
                    temp_fake = generate_fake(netG, Noisee, ii, eva_size)
                    fake = torch.cat((fake, temp_fake), dim=0)

                # print(fake.shape)

                print('-' * 10 + 'Evaluation Begin' + '-' * 10)
                print('----IS-----')
                IS, IS_std = get_inception_score(fake)
                print('Inception Score: {:.2f} +/- {:.2f}'.format(IS, IS_std))
                IS_record.append(IS)
                IS_std_record.append(IS_std)

                torch.cuda.empty_cache()
                print('----FID----')
                fid = eva_fid(cur_epoch)
                fid_record.append(fid)

                with open('./score_record_%s.txt' % model_name, 'a') as f:

                    f.write("epoch " + str(cur_epoch) + ":\n")
                    f.write("FID score:" + str(fid) + '\n')
                    f.write("IS :" + str(IS) + ' (' + str(IS_std) + ')' + '\n')

            # do checkpointing
            torch.save(netG.state_dict(), '%s/netG_%s_epoch_%d.pth' % ('./models', model_name, cur_epoch))
            torch.save(netD.state_dict(), '%s/netD_%s_epoch_%d.pth' % ('./models', model_name, cur_epoch))

    with open('./score_record_%s.txt' % model_name, 'a') as f:
        f.write('\n')
        f.write("Best IS is:" + str(max(IS_record)) + '\n')
        f.write("Best FID is:" + str(min(fid_record)) + '\n')




if __name__ == '__main__':
    model_name = 'WGAN_GP'
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, default=20)
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--lr', type=float, default=0.0001)
    argparser.add_argument('--beta1', type=float, default=0.5)
    argparser.add_argument('--workers', type=int, default=2)
    argparser.add_argument('--image_size', type=int, default=64)
    argparser.add_argument('--noise_size', type=int, default=100)
    argparser.add_argument('--ngf', type=int, default=64)
    argparser.add_argument('--ndf', type=int, default=64)
    argparser.add_argument('--dataroot', type=str, default='../dataset')
    argparser.add_argument('--modelroot', type=str, default='./models')
    argparser.add_argument('--evaluationroot', type=str, default='./evaluation')
    argparser.add_argument('--resultroot', type=str, default='./result')
    argparser.add_argument('--lambda_gp', type=int, default=10)
    argparser.add_argument('--n_critic', type=int, default=5)
    argparser.add_argument('--load', type=bool, default=False)
    argparser.add_argument('--load_epoch', type=int, default=0)

    args = argparser.parse_args()

    # Set random seed for reproducibility
    # manualSeed = 42
    ##
    manualSeed = 12345  # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    if not os.path.exists(args.dataroot):
        os.makedirs(args.dataroot)
    if not os.path.exists(args.modelroot):
        os.makedirs(args.modelroot)
    if not os.path.exists(args.evaluationroot):
        os.makedirs(args.evaluationroot)
    if not os.path.exists(args.resultroot):
        os.makedirs(args.resultroot)
    ngpu = 1
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    main()


