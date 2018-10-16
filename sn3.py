"""""""""
Pytorch implementation of Conditional Image Synthesis with Auxiliary Classifier GANs (https://arxiv.org/pdf/1610.09585.pdf).
This code is based on Deep Convolutional Generative Adversarial Networks in Pytorch examples : https://github.com/pytorch/examples/tree/master/dcgan
"""""""""
from __future__ import print_function
import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import sn_m
import pre_resnet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | mnist')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=50000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

#number of updates to discriminator for every update to generator 
# disc_iters = 1

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


if opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
elif opt.dataset == 'folder':
    # folder dataset
    dataset = dset.ImageFolder(
        root=opt.dataroot,
        transform=transforms.Compose([
            transforms.Resize(opt.imageSize),
            transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True, train=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ])
    )


assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
if opt.dataset == 'mnist':
    nc = 1
    num_classes = 10
if opt.dataset == 'folder':
    nc = 3
    num_classes = 7
else:
    nc = 3
    num_classes = 10

netG = sn_m.netG(nz, ngf, nc)

# if opt.netG != '':
#     netG.load_state_dict(torch.load(opt.netG))
# print(netG)

netD = sn_m.netD(ndf, nc, num_classes)
netD2, D2_feature = pre_resnet.build_model()

# if opt.netD != '':
#     netD.load_state_dict(torch.load(opt.netD))
# print(netD)


netG.load_state_dict(torch.load('netG_epoch_49999.pth'))
netD.load_state_dict(torch.load('netD_epoch_49999.pth'))

dis_criterion = nn.BCELoss()
aux_criterion = nn.NLLLoss()
res_criterion = nn.NLLLoss()    #

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input2 = torch.FloatTensor(opt.batchSize, 3, 224, 224)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
dis_label = torch.FloatTensor(opt.batchSize)
aux_label = torch.LongTensor(opt.batchSize)

real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    dis_criterion.cuda()
    aux_criterion.cuda()
    res_criterion.cuda()    #
    input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()
    input2 = input2.cuda()  #
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
input2 = Variable(input2)   #
dis_label = Variable(dis_label)
aux_label = Variable(aux_label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)
fixed_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))
# random_label = np.random.randint(0, num_classes, opt.batchSize)
random_label = np.zeros((opt.batchSize), dtype=int)
print('fixed label:{}'.format(random_label))
random_onehot = np.zeros((opt.batchSize, num_classes))
random_onehot[np.arange(opt.batchSize), random_label] = 1
fixed_noise_[np.arange(opt.batchSize), :num_classes] = random_onehot[np.arange(opt.batchSize)]
fixed_noise_ = (torch.from_numpy(fixed_noise_))
fixed_noise_ = fixed_noise_.resize_(opt.batchSize, nz, 1, 1)
fixed_noise.data.copy_(fixed_noise_)

# setup optimizer
# optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerRes = optim.Adam(netD2.classifier.parameters(), lr=opt.lr)

def test(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    return correct, len(labels.data)


count = 0

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        img, label = data
        batch_size = img.size(0)

        
        input.data.resize_(img.size()).copy_(img)
        input2.data.resize_(img.size()).copy_(img)
        dis_label.data.resize_(batch_size).fill_(real_label)
        aux_label.data.resize_(batch_size).copy_(label)

        # for _ in range(disc_iters):
        ###########################
        # (1) Update D network
        ###########################
        # train with real
        if(count == 3):
            count = 0
        
        if(count == 2):
            netD.zero_grad()
            netD2.zero_grad()   #
        
        
        dis_output, aux_output = netD(input)
        res_output = netD2(input2)  #
        dis_errD_real = dis_criterion(dis_output, dis_label)
        aux_errD_real = aux_criterion(aux_output, aux_label)
        res_errD_real = res_criterion(res_output, aux_label)  #

        errD_real = dis_errD_real + aux_errD_real + res_errD_real  #
        if(count == 2):
            errD_real.backward()
        D_x = dis_output.data.mean()
        
        correct, length = test(aux_output, aux_label)

        # train with fake
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)

        label = np.random.randint(0, num_classes, batch_size)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        label_onehot = np.zeros((batch_size, num_classes))
        label_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :num_classes] = label_onehot[np.arange(batch_size)]
        
        noise_ = (torch.from_numpy(noise_))
        noise_ = noise_.resize_(batch_size, nz, 1, 1)
        noise.data.copy_(noise_)

        aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

        fake = netG(noise)

        #
        transform1 = transforms.Compose([
            transforms.Resize(opt.imageSize),
            transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        fake2 = transform1(fake.detach())
        # 


        dis_label.data.fill_(fake_label)
        dis_output,aux_output = netD(fake.detach())
        res_output = netD2(fake2)  #


        dis_errD_fake = dis_criterion(dis_output, dis_label)
        aux_errD_fake = aux_criterion(aux_output, aux_label)
        res_errD_fake = res_criterion(res_output, aux_label)  #

        errD_fake = dis_errD_fake + aux_errD_fake + res_errD_fake   #
        if(count == 2):
            errD_fake.backward()

        D_G_z1 = dis_output.data.mean()
        errD = dis_errD_real + dis_errD_fake
        if(count == 2):
            optimizerD.step()
            optimizerRes.step()
            
        count = count + 1
        ###########################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        dis_label.data.fill_(real_label)  # fake labels are real for generator cost
        
        #
        transform1 = transforms.Compose([
            transforms.Resize(opt.imageSize),
            transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        fake2 = transform1(fake.detach())
        # 

        dis_output,aux_output = netD(fake)
        res_output = netD2(fake2)  #

        dis_errG = dis_criterion(dis_output, dis_label)
        aux_errG = aux_criterion(aux_output, aux_label)
        res_errG = res_criterion(res_output, aux_label)  #
        
        errG = dis_errG + aux_errG + res_errG
        errG.backward()
        D_G_z2 = dis_output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f, Accuracy: %.4f / %.4f = %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2,
                 correct, length, 100.* correct / length))

        

    if epoch in [9999,19999,29999,39999,49999]:
        vutils.save_image(img,
                '%s/real_samples.png' % opt.outf)
        #fake = netG(fixed_cat)
        fake = netG(fixed_noise)
        vutils.save_image(fake.data,
                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch + 10000))
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch + 10000))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch + 10000))