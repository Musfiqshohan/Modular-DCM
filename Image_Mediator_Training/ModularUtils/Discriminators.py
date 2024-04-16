import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import pickle
import torch.nn as nn
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional


# Discriminator model

class ControllerDiscriminator(nn.Module):

    def __init__(self, Exp, pac=10, **kwargs):
        super(ControllerDiscriminator, self).__init__()

        input_dim = kwargs['input_dim']

        input_dim = input_dim * pac
        self.pac = pac
        self.pacdim = input_dim

        output_dim = 1
        hidden_dims = Exp.D_hid_dims

        print(f'Critic init: indim {input_dim}  outdim {output_dim}')

        # self.input_layer = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dims[0]),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(0.3)
        # )
        #
        # self.hidden_layers = nn.ModuleList()
        # for i in range(len(hidden_dims)-1):
        #     hid_layer = nn.Sequential(
        #         nn.Linear(hidden_dims[i], hidden_dims[i+1]),
        #         nn.LeakyReLU(0.2),
        #         nn.Dropout(0.3)
        #     )
        #     self.hidden_layers.append(hid_layer)
        #
        #
        # self.output_layer = nn.Sequential(
        #     nn.Linear(hidden_dims[-1], output_dim),
        #     # nn.Sigmoid()    #Doesnt use the sigmoid function in WGAN
        # )

        dim = input_dim
        seq = []
        for item in list(hidden_dims):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def forward(self, generated_data):
        assert generated_data.size()[0] % self.pac == 0
        generated_data = generated_data.view(-1, self.pacdim)

        input = generated_data
        # output = self.input_layer(input)
        # for i in range(len(self.hidden_layers)):
        #     output= self.hidden_layers[i](output)
        # output = self.output_layer(output)

        output = self.seq(input)
        return output


class DigitImageDiscriminator(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DigitImageDiscriminator, self).__init__()

        self.image_dim = kwargs['image_dim']
        self.label_dims = kwargs['label_dims']
        num_filters = kwargs['num_filters']
        self.output_dim = kwargs['output_dim']

        print(f'Critic init: image dim {self.image_dim} compare dim {self.label_dims}  outdim {self.output_dim}')

        self.hidden_layer1 = torch.nn.Sequential()
        self.hidden_layer2 = torch.nn.Sequential()
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Convolutional layer
            if i == 0:
                # For input
                input_conv = torch.nn.Conv2d(self.image_dim, int(num_filters[i] / 2), kernel_size=4, stride=2,
                                             padding=1)
                # input_conv = torch.nn.Conv2d(self.image_dim, int(num_filters[i]/2), kernel_size=3, stride=1, padding=1)

                self.hidden_layer1.add_module('input_conv', input_conv)

                # Activation
                self.hidden_layer1.add_module('input_act', torch.nn.LeakyReLU(0.2))

                # For label
                label_conv = torch.nn.Conv2d(self.label_dims, int(num_filters[i] / 2), kernel_size=4, stride=2,
                                             padding=1)
                # label_conv = torch.nn.Conv2d(self.label_dims, int(num_filters[i]/2), kernel_size=3, stride=1, padding=1)
                self.hidden_layer2.add_module('label_conv', label_conv)

                # Activation
                self.hidden_layer2.add_module('label_act', torch.nn.LeakyReLU(0.2))
            else:
                conv = torch.nn.Conv2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=1,
                                       bias=False)

                conv_name = 'conv' + str(i + 1)
                self.hidden_layer.add_module(conv_name, conv)

                # instance norm normalization
                in_name = 'in' + str(i + 1)
                self.hidden_layer.add_module(in_name, torch.nn.InstanceNorm2d(num_filters[i], affine=True))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Convolutional layer
        out = torch.nn.Conv2d(num_filters[i], self.output_dim, kernel_size=4, stride=1, padding=0)
        self.output_layer.add_module('out', out)

    def forward(self, image, parents):
        h1 = self.hidden_layer1(image)
        h2 = self.hidden_layer2(parents)
        x = torch.cat([h1, h2], 1)
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out
