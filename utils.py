import os
import torch
from torch import nn
from torch.autograd import Variable, grad
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import torchvision.utils as vutils
import numpy as np
import torch.autograd as autograd


def eva_fid(epoch):
    # print('----Start computing FID----')
    temp = os.popen('python -m pytorch_fid ./evaluation ../dataset/cifar10.test.npz').readlines()
    # print(type(temp))
    # print(temp)
    temp = temp[0].replace('\n\'', '')
    temp = temp.replace('\'', '')
    temp = temp.replace('FID:  ', '')
    temp = float(temp)
    print('Epoch: %d FID: %f'%(epoch, temp))
    return temp


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

        
def generate_fake(netG, noise, ii, eva_size=2000):
    fake_temp = netG(noise)
    for i in range(eva_size):
        vutils.save_image(fake_temp[i].detach(), './evaluation/random_%s.png' % (str((ii + 1) * eva_size + i)),
                          normalize=True)
    return fake_temp




def compute_gradient_penalty(discriminator, real_data, fake_data):
    alpha = torch.cuda.FloatTensor(fake_data.shape[0], 1, 1, 1).uniform_(0, 1).expand(fake_data.shape)
    interpolates = alpha * fake_data + (1 - alpha) * real_data
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = discriminator(interpolates)

    grad_out = torch.ones(disc_interpolates.size()).cuda()
    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs = grad_out,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


