import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional


def sample_gumbel(Exp, shape, eps=1e-20):
    U = torch.rand(shape).to(Exp.DEVICE)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(Exp, logits, temperature, gumbel_noise):

    if gumbel_noise==None:
        gumbel_noise= sample_gumbel(Exp, logits.size())

    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(Exp, logits, temperature, gumbel_noise=None, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    output_dim =logits.shape[1]
    y = gumbel_softmax_sample(Exp, logits, temperature, gumbel_noise)

    if not hard:
        ret = y.view(-1, output_dim)
        return ret

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    # ret = y_hard.view(-1, output_dim)
    ret = y_hard.view(-1, output_dim)
    return ret

# def _gumbel_softmax(logits, tau=1,  dim=-1, hard=False, eps=1e-10):
#     """Deals with the instability of the gumbel_softmax for older versions of torch.
#
#     For more details about the issue:
#     https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
#
#     Args:
#         logits […, num_features]:
#             Unnormalized log probabilities
#         tau:
#             Non-negative scalar temperature
#         hard (bool):
#             If True, the returned samples will be discretized as one-hot vectors,
#             but will be differentiated as if it is the soft sample in autograd
#         dim (int):
#             A dimension along which softmax will be computed. Default: -1.
#
#     Returns:
#         Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
#     """
#     if version.parse(torch.__version__) < version.parse('1.2.0'):
#         for i in range(10):
#             transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
#                                                     eps=eps, dim=dim)
#             if not torch.isnan(transformed).any():
#                 return transformed
#         raise ValueError('gumbel_softmax returning NaN.')
#
#     return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

class Residual(Module):
    """Residual layer for the CTGANSynthesizer."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)



class ControllerGenerator(torch.nn.Module):

    def __init__(self, Exp, **kwargs):
        super(ControllerGenerator, self).__init__()
        self.hidden_dims = Exp.G_hid_dims
        self.input_dim = kwargs['input_dim']

        self.output_dim = kwargs['output_dim']

        print(f'Causal Generator init: indim {self.input_dim}  outdim {self.output_dim}')

        # self.input_layer = nn.Sequential(
        #     nn.Linear(self.input_dim, self.hidden_dims[0]),
        #     nn.LeakyReLU(0.2)
        # )
        #
        # self.hidden_layers = nn.ModuleList()
        #
        # for i in range(len(self.hidden_dims) - 1):
        #     hid_layer = nn.Sequential(
        #         nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
        #         nn.LeakyReLU(0.2)
        #     )
        #     self.hidden_layers.append(hid_layer)
        #
        # self.output_layer = nn.Sequential(
        #     nn.Linear(self.hidden_dims[-1], self.output_dim),
        #     # nn.Sigmoid()  #following my previous implementation
        # )

        dim = self.input_dim
        seq = []
        for item in list(self.hidden_dims):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, self.output_dim))
        self.seq = Sequential(*seq)




    def forward(self, Exp, noise, gen_labels, **kwargs):

        input = torch.cat(noise, 1)
        if len(gen_labels) > 0:
            gen_labels = torch.cat(gen_labels, 1)
            # gen_labels = self.input_label_layer(gen_labels)
            input = torch.cat([input, gen_labels], 1)


        # output = self.input_layer(input)
        # for i in range(len(self.hidden_layers)):
        #     output = self.hidden_layers[i](output)
        # output = self.output_layer(output)

        output = self.seq(input)


        output = output[:, 0: self.output_dim]


        # output_feature = _gumbel_softmax(output, Exp.Temperature).to(Exp.DEVICE)
        output_feature = gumbel_softmax(Exp, output, Exp.Temperature, kwargs["gumbel_noise"], kwargs["hard"]).to(Exp.DEVICE)


        final_output = output_feature
        return final_output


class DigitImageGenerator(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DigitImageGenerator, self).__init__()

        self.noise_dim = kwargs['noise_dim']
        self.parent_dims = kwargs['parent_dims']

        num_filters= kwargs['num_filters']
        self.output_dim = kwargs['output_dim']


        print(f'Image Generator init: noise dim: {self.noise_dim}, parent dim:{self.parent_dims}  outdim {self.output_dim}')


        # Hidden layers
        self.hidden_layer1 = torch.nn.Sequential()
        self.hidden_layer2 = torch.nn.Sequential()
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Deconvolutional layer
            if i == 0:
                # For input
                input_deconv = torch.nn.ConvTranspose2d(self.noise_dim, int(num_filters[i] / 2), kernel_size=4, stride=1,padding=0)
                self.hidden_layer1.add_module('input_deconv', input_deconv)

                # Batch normalization
                self.hidden_layer1.add_module('input_bn', torch.nn.BatchNorm2d(int(num_filters[i] / 2)))

                # Activation
                self.hidden_layer1.add_module('input_act', torch.nn.ReLU())

                # For label
                label_deconv = torch.nn.ConvTranspose2d( self.parent_dims , int(num_filters[i] / 2), kernel_size=4,stride=1, padding=0)
                self.hidden_layer2.add_module('label_deconv', label_deconv)

                # Batch normalization
                self.hidden_layer2.add_module('label_bn', torch.nn.BatchNorm2d(int(num_filters[i] / 2)))

                # Activation
                self.hidden_layer2.add_module('label_act', torch.nn.ReLU())
            else:
                deconv = torch.nn.ConvTranspose2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2,padding=1)

                deconv_name = 'deconv' + str(i + 1)
                self.hidden_layer.add_module(deconv_name, deconv)

                # Batch normalization
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer.add_module(act_name, torch.nn.ReLU())

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Deconvolutional layer
        out = torch.nn.ConvTranspose2d(num_filters[i], self.output_dim, kernel_size=4, stride=2, padding=1)
        # out = torch.nn.ConvTranspose2d(num_filters[i], self.output_dim, kernel_size=3, stride=1, padding=1)  #if we want 32x32 for filter [256, 128, 64, 32]
        self.output_layer.add_module('out', out)

        # Activation
        self.output_layer.add_module('act', torch.nn.Tanh())

    def forward(self, noise, gen_labels):
        noises = torch.cat(noise, 1)
        gen_labels = torch.cat(gen_labels, 1)  # there will be always some parent to the digit image, otherwise gen_labels is just noise.
        gen_labels=  gen_labels.view(-1, gen_labels.shape[1], 1, 1)

        h1 = self.hidden_layer1(noises)
        h2 = self.hidden_layer2(gen_labels)
        x = torch.cat([h1, h2], 1)
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out



# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class ClassificationNet(nn.Module):
    def __init__(self, **kwargs):
        super(ClassificationNet, self).__init__()
        self.output_dim = kwargs['output_dim']
        self.parent_dims = kwargs['parent_dims']
        self.image_dim = kwargs['image_dim']

        # self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, self.output_dim)

        self.conv1 = nn.Conv2d(self.image_dim+self.parent_dims, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,self.output_dim)

    def forward(self, Exp, x, **kwargs):
        # x= torch.cat(x, 1)

        # x1 = self.conv1(x)
        # x2 =F.max_pool2d(x1, 2)
        # x = F.relu(x2)
        # # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)


        #-------
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # return x

        output_feature = gumbel_softmax(Exp, x, Exp.Temperature, kwargs["gumbel_noise"], kwargs["hard"]).to(Exp.DEVICE)

        return output_feature


class ConditionalClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(ConditionalClassifier, self).__init__()
        self.output_dim = kwargs['output_dim']
        cond_dim=kwargs['parent_dims']

        self.conv1 = nn.Conv2d(3+cond_dim, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,self.output_dim)

    def forward(self, Exp, noise, image,  **kwargs):  #condition or noise
        noise = torch.cat(noise, 1)
        x= torch.cat([noise, image], 1)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        output_feature = gumbel_softmax(Exp, x, Exp.Temperature).to(Exp.DEVICE)

        return output_feature





# mish activation function
class mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        mish()
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        mish(),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        mish()
    )


class conditionalMobileNet(nn.Module):
    def __init__(self, **kwargs):
        super(conditionalMobileNet, self).__init__()

        # input_dim = 3 + kwargs['exos_dim'] + kwargs['conf_dim']  #exogenous and confounding noise
        input_dim = 3 + kwargs['exos_dim']   #exogenous and confounding noise
        output_dim = kwargs['output_dim']

        print("input_dim",input_dim)

        # num_labels = 3
        self.features = nn.Sequential(
            conv_bn(input_dim, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )

        self.fc = nn.Linear(1024, output_dim)

    def forward(self, Exp, noises, x, **kwargs):
        noises = torch.cat(noises, 1)
        x = torch.cat([noises, x], 1)

        print('Xshape', x.shape)

        # x = torch.cat(x, 1)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        output_feature = gumbel_softmax(Exp, x, Exp.Temperature, kwargs["gumbel_noise"], kwargs["hard"]).to(Exp.DEVICE)
        return output_feature