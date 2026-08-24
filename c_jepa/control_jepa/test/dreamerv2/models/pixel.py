import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F


class ObsEncoder(nn.Module):
    def __init__(self, input_shape, embedding_size, info):
        """
        :param input_shape: tuple containing shape of input
        :param embedding_size: Supposed length of encoded vector
        """
        super(ObsEncoder, self).__init__()
        self.shape = input_shape
        activation = info['activation']
        d = info['depth']
        k  = info['kernel']
        s = info['stride'] if 'stride' in info else [1, 1, 1]
        self.k = k
        self.d = d
        self.s = s
        self.convolutions = nn.Sequential(
            nn.Conv2d(input_shape[0], d, k[0],stride=s[0]),   #nn.Conv2d(input_shape[0], d, k), edit josh
            activation(),
            nn.Conv2d(d, 2*d, k[1],stride=s[1]),              #nn.Conv2d(d, 2*d, k), edit josh
            activation(),
            nn.Conv2d(2*d, 4*d, k[2],stride=s[2]),            #nn.Conv2d(2*d, 4*d, k), edit josh
            activation(),
        )
        if embedding_size == self.embed_size:
            self.fc_1 = nn.Identity()
        else:
            self.fc_1 = nn.Linear(self.embed_size, embedding_size)

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        embed = self.convolutions(obs.reshape(-1, *img_shape))
        embed = torch.reshape(embed, (*batch_shape, -1))
        embed = self.fc_1(embed)
        return embed

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.shape[1:], 0, self.k[0], self.s[0])
        conv2_shape = conv_out_shape(conv1_shape, 0, self.k[1], self.s[1])
        conv3_shape = conv_out_shape(conv2_shape, 0, self.k[2], self.s[2])
        embed_size = int(4*self.d*np.prod(conv3_shape).item())
        return embed_size

class ObsDecoder(nn.Module):
    def __init__(self, output_shape, embed_size, info):
        """
        :param output_shape: tuple containing shape of output obs
        :param embed_size: the size of input vector, for dreamerv2 : modelstate 
        """
        super(ObsDecoder, self).__init__()
        c, h, w = output_shape
        activation = info['activation']
        d = info['depth']
        k  = info['kernel']
        s  = info['stride'] if 'stride' in info else [1, 1, 1] 
        conv1_shape = conv_out_shape(output_shape[1:], 0, k[0], stride=s[0])
        conv2_shape = conv_out_shape(conv1_shape, 0, k[1], stride=s[1])
        conv3_shape = conv_out_shape(conv2_shape, 0, k[2], stride=s[2])
        self.conv_shape = (4*d, *conv3_shape)
        self.output_shape = output_shape
        if embed_size == np.prod(self.conv_shape).item():
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(embed_size, np.prod(self.conv_shape).item())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4*d, 2*d, k[0], stride=s[0],padding=1),
            activation(),
            nn.ConvTranspose2d(2*d, d, k[1], stride=s[1],padding=1),
            activation(),
            nn.ConvTranspose2d(d, c, k[2], stride=s[2],padding=0,output_padding=2),
        )

    def forward(self, x):
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)
        x = self.linear(x)
        x = torch.reshape(x, (squeezed_size, *self.conv_shape))
        x = self.decoder(x)

        x = F.interpolate(x, size=self.output_shape[1:], mode="bilinear", align_corners=False)
        mean = torch.reshape(x, (*batch_shape, *self.output_shape))
        obs_dist = td.Independent(td.Normal(mean, 1), len(self.output_shape))
        return obs_dist
    
def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)

def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))
