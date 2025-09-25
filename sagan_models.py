import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())
        
        curr_dim = int(curr_dim / 2)
        ch_after_l3 = curr_dim

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)
                    
        # for >=128
        if self.imsize >= 128:
            layer5 = []
            layer5.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer5.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer5.append(nn.ReLU())
            self.l5 = nn.Sequential(*layer5)
            curr_dim = int(curr_dim / 2)

        if self.imsize == 256:
            layer6 = []
            layer6.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer6.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer6.append(nn.ReLU())
            self.l6 = nn.Sequential(*layer6)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        # Initialize attention with the exact channel sizes where they will run
        self.attn1 = Self_Attn(ch_after_l3, 'relu')  # runs right after l3
        self.attn2 = Self_Attn(curr_dim, 'relu')     # runs after l4/l5/l6

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

    # --- in Generator.forward ---
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.l1(z)
        out = self.l2(out)
        out = self.l3(out)

        # REMOVE this line (it causes the mismatch and optimizer issues):
        # self.attn1 = Self_Attn(512, 'relu')

        out, p1 = self.attn1(out)

        if hasattr(self, 'l4'):
            out = self.l4(out)
        if hasattr(self, 'l5'):
            out = self.l5(out)
        if hasattr(self, 'l6'):
            out = self.l6(out)

        out, p2 = self.attn2(out)
        out = self.last(out)
        return out, p1, p2

class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        
        if self.imsize >= 128:
            layer5 = []
            layer5.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim, 4, 2, 1)))  # down one more
            layer5.append(nn.LeakyReLU(0.1))
            self.l5 = nn.Sequential(*layer5)

        if self.imsize == 256:
            layer6 = []
            layer6.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim, 4, 2, 1)))
            layer6.append(nn.LeakyReLU(0.1))
            self.l6 = nn.Sequential(*layer6)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        
        self.attn1 = Self_Attn(curr_dim, 'relu')

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn2 = Self_Attn(curr_dim, 'relu')


    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)

        # attention 1 at current channels
        out, p1 = self.attn1(out)

        # optional downsamples (only if they exist)
        if hasattr(self, 'l4'):
            out = self.l4(out)
        if hasattr(self, 'l5'):
            out = self.l5(out)
        if hasattr(self, 'l6'):
            out = self.l6(out)

        # attention 2 at current channels
        out, p2 = self.attn2(out)

        # make sure spatial is 4x4 for the final conv (safe for all sizes)
        if hasattr(self, 'pool4'):
            out = self.pool4(out)  # nn.AdaptiveAvgPool2d(4)

        out = self.last(out)       # nn.Conv2d(curr_dim, 1, 4)
        return out.squeeze(), p1, p2

