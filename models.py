# pyTorch=1.9.1+cu111

import torch.nn as nn
import torch

class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        return (x-torch.mean(x))/torch.sqrt(torch.var(x)+self.eps)

class D_Block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.conv1 = nn.Conv3d(dim_in, dim_out, 3, 2, 1)
        self.norm = InstanceNorm()
        self.actv = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.actv(x)

        return x


class GMT_D(nn.Module):
    def __init__(self, conv_dim=32):
        super(GMT_D, self).__init__()

        self.convs = nn.ModuleList([
            D_Block(1, conv_dim),
            D_Block(conv_dim, conv_dim*2),
            D_Block(conv_dim*2, conv_dim*4),
            D_Block(conv_dim*4, conv_dim*8),
            D_Block(conv_dim*8, conv_dim*8),
        ])

        self.fcs = nn.Sequential(
            nn.Linear(conv_dim*8*2*2*2,512), nn.LeakyReLU(0.2),
            nn.Linear(512,512), nn.LeakyReLU(0.2),
            nn.Linear(512,1), nn.Sigmoid()
        )

    def forward(self, x):
        feats = []

        for i, conv in enumerate(self.convs):
            x = conv(x)

            if i < 3:
                    feats.append(x)

        x = x.view(x.size(0),-1)

        x = self.fcs(x)

        return x, feats

class MLP(nn.Module):
    def __init__(self, c_dim):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(c_dim,512), nn.LeakyReLU(0.2),
            nn.Linear(512,512), nn.LeakyReLU(0.2),
            nn.Linear(512,512), nn.LeakyReLU(0.2),
            nn.Linear(512,512), nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        s = self.layers(x)
        return s

class EncBlk(nn.Module):
    def __init__(self, dim_in, dim_out, downsample=True, padding_mode='reflect'):
        super().__init__()

        if downsample:
            self.conv1 = nn.Conv3d(dim_in,dim_out,3,2,1,padding_mode=padding_mode)
        else:
            self.conv1 = nn.Conv3d(dim_in,dim_out,3,1,1,padding_mode=padding_mode)
        
        self.conv2 = nn.Conv3d(dim_out,dim_out,3,1,1,padding_mode=padding_mode)

        self.norm = InstanceNorm()
        self.actv = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.actv(x)

        x = self.conv2(x)
        x = self.norm(x)
        x = self.actv(x)

        return x

class DecBlk(nn.Module):
    def __init__(self, dim_in, dim_out, padding_mode='reflect'):
        super().__init__()

        self.upconv = nn.ConvTranspose3d(dim_in, dim_out, 2, 2)

        self.conv1 = nn.Conv3d(dim_in,dim_out,3,1,1,padding_mode=padding_mode)
        self.conv2 = nn.Conv3d(dim_out,dim_out,3,1,1,padding_mode=padding_mode)

        self.fc1 = nn.Linear(512,dim_out*2)
        self.fc2 = nn.Linear(512,dim_out*2)

        self.actv = nn.LeakyReLU(0.2, inplace=True)
        self.norm = InstanceNorm()

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear")

    def adain(self, x, s):
        w, b = s.unsqueeze(2).unsqueeze(3).unsqueeze(4).chunk(2, 1)
        return x * w + b

    def forward(self, x, conn, s):
        x = self.upconv(x)

        x = torch.cat((conn,x), dim=1)

        x = self.conv1(x)
        x = self.norm(x)
        x = self.adain(x, self.fc1(s))
        x = self.actv(x)
    
        x = self.conv2(x)
        x = self.norm(x)
        x = self.adain(x, self.fc2(s))
        x = self.actv(x)

        return x



class GMT(nn.Module):
    def __init__(self, c_dim=2):
        super(GMT, self).__init__()

        self.mlp = MLP(c_dim)

        self.encoder = nn.ModuleList([
            EncBlk(1,32,downsample=False),
            # EncBlk(16,32),
            EncBlk(32,64),
            EncBlk(64,128),
            EncBlk(128,256),
            EncBlk(256,512),
        ])

        self.decoder = nn.ModuleList([
            DecBlk(512,256),
            DecBlk(256,128),
            DecBlk(128,64),
            DecBlk(64,32),
            # DecBlk(32,16),
        ])

        self.final_conv = nn.Conv3d(32,1,1)

        self.tanh = nn.Tanh()

    def forward(self, x, c):
        s = self.mlp(c.view(1,-1))

        e = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            e.append(x)

        for i, block in enumerate(self.decoder):
            x = block(x, e[len(self.encoder)-2-i],s)

        x = self.final_conv(x)
        x = self.tanh(x)
        return x

