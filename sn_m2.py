import torch.nn as nn
from spectral_normalization import SpectralNorm


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class netG(nn.Module):

    def __init__(self, nz, ngf, nc):

        super(netG, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.Tanh = nn.Tanh()
        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(ngf * 8)

        self.conv21 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False)
        self.BatchNorm21 = nn.BatchNorm2d(ngf * 8)

        self.conv22 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False)
        self.BatchNorm22 = nn.BatchNorm2d(ngf * 8)

        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ngf * 4)

        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ngf * 2)

        self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(ngf * 1)

        self.conv5 = nn.ConvTranspose2d(ngf * 1, nc, 4, 2, 1, bias=False)
        
        self.apply(weights_init)


    def forward(self, input):

        x = self.conv1(input)
        x = self.BatchNorm1(x)
        x = self.ReLU(x)
        print(x.shape)
        x = self.conv21(x)
        x = self.BatchNorm21(x)
        x = self.ReLU(x)
        print(x.shape)
        x = self.conv21(x)
        x = self.BatchNorm22(x)
        x = self.ReLU(x)
        print(x.shape)
        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.ReLU(x)
        print(x.shape)
        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.ReLU(x)
        print(x.shape)
        x = self.conv4(x)
        x = self.BatchNorm4(x)
        x = self.ReLU(x)
        print(x.shape)

        x = self.conv5(x)
        print(x.shape)
        output = self.Tanh(x)
        return output

    
class netD(nn.Module):

    def __init__(self, ndf, nc, nb_label):

        super(netD, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = SpectralNorm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        self.conv2 = SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        # self.BatchNorm2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        # self.BatchNorm3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))
        # self.BatchNorm4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = SpectralNorm(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False))
        # self.BatchNorm4 = nn.BatchNorm2d(ndf * 8)
        self.conv6 = SpectralNorm(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False))
        # self.BatchNorm4 = nn.BatchNorm2d(ndf * 8)

        self.conv7 = SpectralNorm(nn.Conv2d(ndf * 8, ndf * 1, 4, 1, 0, bias=False))
        self.disc_linear = nn.Linear(ndf * 1, 1)
        self.aux_linear = nn.Linear(ndf * 1, nb_label)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.ndf = ndf
        # self.apply(weights_init)

    def forward(self, input):
        print(input.shape)

        x = self.conv1(input)
        x = self.LeakyReLU(x)
        # print(x.shape)

        x = self.conv2(x)
        # x = self.BatchNorm2(x)
        x = self.LeakyReLU(x)
        # print(x.shape)

        x = self.conv3(x)
        # x = self.BatchNorm3(x)
        x = self.LeakyReLU(x)
        # print(x.shape)

        x = self.conv4(x)
        # x = self.BatchNorm4(x)
        x = self.LeakyReLU(x)
        # print(x.shape)


        x = self.conv5(x)
        # x = self.BatchNorm4(x)
        x = self.LeakyReLU(x)
        # print(x.shape)
        x = self.conv6(x)
        # x = self.BatchNorm4(x)
        x = self.LeakyReLU(x)
        # print(x.shape)

        x = self.conv7(x)
        # print(x.shape)

        x = x.view(-1, self.ndf)
        # print(x.shape)

        c = self.aux_linear(x)
        classes = self.softmax(c)
        s = self.disc_linear(x)
        
        realfake = self.sigmoid(s)

        # classes = self.softmax(c)
        # realfake = self.sigmoid(s).view(-1, 1).squeeze(1)
        # print(realfake.shape)
        # return s,c
        return realfake, classes