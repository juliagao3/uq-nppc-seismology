import torch
import torch.nn as nn

from eikonal_restoration import *


class UNet(nn.Module):
    def __init__(
        self, 
        in_channels=1, 
        out_channels=1,
        channels_list=(32, 64, 128, 256),
        bottleneck_channels=512,
        min_channels_decoder=64,
        n_groups=8,
    ):
    
        super().__init__()
        ch = in_channels
        
        #encoder
        self.encoder_blocks = nn.ModuleList()
        ch_hidden_list = []
        
        layers = []
        layers.append(nn.ZeroPad2d(2))
        ch_ = channels_list[0]
        layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
        ch = ch_
        self.encoder_blocks.append(nn.Sequential(*layers))
        ch_hidden_list.append(ch)

        for i_level in range(len(channels_list)):
            ch_ = channels_list[i_level]
            downsample = i_level != 0

            layers = []
            if downsample:
                layers.append(nn.MaxPool2d(2))
            layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
            ch = ch_
            layers.append(nn.GroupNorm(n_groups, ch))
            layers.append(nn.LeakyReLU(0.1))
            self.encoder_blocks.append(nn.Sequential(*layers))
            ch_hidden_list.append(ch)

        ## Bottleneck
        ## ==========
        ch_ = bottleneck_channels
        layers = []
        layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
        ch = ch_
        layers.append(nn.GroupNorm(n_groups, ch))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Conv2d(ch, ch, 3, padding=1))
        layers.append(nn.GroupNorm(n_groups, ch))
        layers.append(nn.LeakyReLU(0.1))
        self.bottleneck = nn.Sequential(*layers)

        ## Decoder
        ## =======
        self.decoder_blocks = nn.ModuleList([])
        for i_level in reversed(range(len(channels_list))):
            ch_ = max(channels_list[i_level], min_channels_decoder)
            downsample = i_level != 0
            ch = ch + ch_hidden_list.pop()
            layers = []

            layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
            ch = ch_
            layers.append(nn.GroupNorm(n_groups, ch))
            layers.append(nn.LeakyReLU(0.1))
            if downsample:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.decoder_blocks.append(nn.Sequential(*layers))

        ch = ch + ch_hidden_list.pop()
        ch_ = channels_list[0]
        layers = []
        layers.append(nn.Conv2d(ch, out_channels, 1))
        layers.append(nn.ZeroPad2d(-2))
        self.decoder_blocks.append(nn.Sequential(*layers))
    
    def forward(self, x):
        h = []
        for block in self.encoder_blocks:
            x = block(x)
            
            h.append(x)

        x = self.bottleneck(x)
        for block in self.decoder_blocks:
            x = torch.cat((x, h.pop()), dim=1)
            x = block(x)
        return x   
    
def gram_schmidt(x):
    x_shape = x.shape
    x = x.flatten(2)

    x_orth = []
    proj_vec_list = []
    for i in range(x.shape[1]):
        w = x[:, i, :]
        for w2 in proj_vec_list:
            w = w - w2 * torch.sum(w * w2, dim=-1, keepdim=True)
        w_hat = w.detach() / w.detach().norm(dim=-1, keepdim=True)

        x_orth.append(w)
        proj_vec_list.append(w_hat)

    x_orth = torch.stack(x_orth, dim=1).view(*x_shape)
    return x_orth

class RestorationWrapper(nn.Module):
    def __init__(self, net, linear_projection_x_dim):
        super().__init__()

        self.net = net
        self.linear_projection_x_dim = linear_projection_x_dim
        self.linear_projection = nn.Linear(self.linear_projection_x_dim * 26, 28 * 28)

    def forward(self, x):  
        x = x.float()
        x = x.view(x.size(0), -1)
        x = self.linear_projection(x)
        x = x.view(x.size(0), 1, 28, 28)

        x = (x - 0.5) / 0.2
        x = self.net(x)
        x = (x * 0.2) + 0.5

        return x


class PCWrapper(nn.Module):
    def __init__(self, net, n_dirs, linear_projection_x_dim):
        super().__init__()

        self.net = net
        self.n_dirs = n_dirs
        self.linear_projection_x_dim = linear_projection_x_dim
        
        self.linear_projection = nn.Linear(self.linear_projection_x_dim * 26, 28 * 28)

    def forward(self, x_distorted, x_restored): 
        x_distorted = x_distorted.float()
        x_distorted = x_distorted.view(x_distorted.size(0), -1)
        x_distorted = self.linear_projection(x_distorted)
        x_distorted = x_distorted.view(x_distorted.size(0), 1, 28, 28)

        x_restored = x_restored.float()
        x = torch.cat((x_distorted, x_restored), dim=1)

        x = (x - 0.5) / 0.2
        
        # print(x.shape)
        w_mat = self.net(x)
        w_mat = w_mat * 0.2

        w_mat = w_mat.unflatten(1, (self.n_dirs, w_mat.shape[1] // self.n_dirs))
        w_mat = w_mat.flatten(0, 1)
        w_mat = w_mat.unflatten(0, (w_mat.shape[0] // self.n_dirs, self.n_dirs))

        w_mat = gram_schmidt(w_mat)
        return w_mat 