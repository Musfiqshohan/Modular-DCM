from ModularUtils import ControllerConstants
from ModularUtils.FunctionsDistribution import get_joint_distributions_from_samples
from ModularUtils.ControllerConstants import map_dictfill_to_discrete, map_fill_to_discrete, get_multiple_labels_fill
from ModularUtils.DigitImageGeneration.mnist_image_generation import produce_uniform_images
from ModularUtils.Discriminators import DigitImageDiscriminator, ControllerDiscriminator
from ModularUtils.Generators import DigitImageGenerator, ControllerGenerator, ClassificationNet, gumbel_softmax
import torch
from pathlib import Path
from numpy import uint8

from torch import optim as optim
from torchvision import transforms

from ModularUtils.FunctionsTraining import get_training_variables
from MNISTVae import DeepAutoencoder
from MNISTVae import CelebAutoencoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional

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

        input_dim= 3+ kwargs['noise_dim']
        output_dim = kwargs['output_dim']

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
        x = torch.cat(x, 1)
        x = torch.cat([noises,x], 1)

        x= torch.cat(x, 1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


        output_feature = gumbel_softmax(Exp, x, Exp.Temperature, kwargs["gumbel_noise"], kwargs["hard"]).to(Exp.DEVICE)
        return output_feature

# from ModularUtils.imageVae import DeepAutoencoder


def get_generators(Exp, load_which_models, gen_check_point):
    label_generators = {}
    optimizersMech = {}

    for label in Exp.Observed_DAG:

        noise_dims = Exp.NOISE_DIM + Exp.CONF_NOISE_DIM * len(
            Exp.latent_conf[label])

        parent_dims = 0
        for par in Exp.Observed_DAG[label]:
            parent_dims += Exp.label_dim[par]

        if label in Exp.image_labels:
            label_generators[label] = DigitImageGenerator(noise_dim=Exp.IMAGE_NOISE_DIM,
                                                          parent_dims=parent_dims,
                                                          num_filters=Exp.IMAGE_FILTERS,
                                                          output_dim= 3).to(Exp.DEVICE)  # mnistImage

            optimizersMech[label] = torch.optim.Adam(label_generators[label].parameters(), lr=Exp.learning_rate,
                                                     betas=Exp.betas, weight_decay=Exp.generator_decay)

        elif label in Exp.rep_labels:
            # Instantiating the model and hyperparameters
            if Exp.DAG_desc =="celebAtrain":
                label_generators[label]= CelebAutoencoder(Exp, parent_dims, latent_dim=Exp.ENCODED_DIM).to(Exp.DEVICE)
            else:
                label_generators[label] = DeepAutoencoder(Exp, parent_dims, image_dim=3, latent_dim=Exp.ENCODED_DIM).to(Exp.DEVICE)

            optimizersMech[label] = torch.optim.Adam(label_generators[label].parameters(), lr=Exp.learning_rate,
                                                     betas=Exp.betas,  weight_decay=Exp.generator_decay)

        elif set(Exp.Observed_DAG[label]) & set(Exp.image_labels) != set():

            if Exp.DAG_desc =="celebAtrain":
                label_generators[label]= conditionalMobileNet(noise_dim=Exp.CONF_NOISE_DIM, output_dim=Exp.label_dim[label]).to(Exp.DEVICE)
                optimizersMech[label] = torch.optim.Adam(label_generators[label].parameters(), lr=Exp.learning_rate, betas=(0.9, 0.999))

            else:
                parent_dims+= noise_dims
                label_generators[label] = ClassificationNet(parent_dims=parent_dims, image_dim=3, output_dim=Exp.label_dim[label]).to(Exp.DEVICE)
                momentum = 0.5
                optimizersMech[label] = optim.SGD(label_generators[label].parameters(), lr=Exp.learning_rate,
                                                  momentum=momentum)

        else:
            label_generators[label] = ControllerGenerator(Exp, input_dim=noise_dims + parent_dims,
                                                          output_dim=Exp.label_dim[label],
                                                          ).to(Exp.DEVICE)  # mnistImage

            optimizersMech[label] = torch.optim.Adam(label_generators[label].parameters(), lr=Exp.learning_rate,
                                                     betas=Exp.betas,  weight_decay=Exp.generator_decay)

    # loading saved generator if required
    if True in load_which_models.values():
        # gfile = Exp.LOAD_MODEL_PATH + "/gen_checkpoints/epochLast.pth"
        checkpointx = torch.load(gen_check_point, map_location="cuda")
        # Exp.checkpoints["generator"]= checkpointx["generator"]


    for lbid, label in enumerate(Exp.label_names):
        if load_which_models[label] == True:
            # last_model= Exp.checkpoints["generator"][-1]
            last_model= checkpointx
            label_generators[label].load_state_dict(last_model[label+"state_dict" ])
            # optimizersMech[label].load_state_dict(last_model[label+"optimizer"])
            optimizersMech[label].load_state_dict(last_model["optimizer"+label])
            for param_group in optimizersMech[label].param_groups:
                param_group["lr"] = Exp.learning_rate

            print(f'{label} generator loaded')
        else:
            label_generators[label].apply(ControllerConstants.init_weights)

    return label_generators, optimizersMech


def get_discriminators(Exp, cur_hnodes, load_which_models, disc_check_point):

    discriminatorsMech={}
    doptimizersMech={}


    # comparedim_list=[] #for each interventional dataset
    for hnode, cur_mechs in cur_hnodes.items():

        for ino, intv in enumerate(Exp.Data_intervs):
            all_compare_Var, compare_Var, intervened_Var, real_labels_vars = get_training_variables(Exp, cur_mechs, ino, intv)

            compare_dims = 0
            for var in real_labels_vars:
                compare_dims += Exp.label_dim[var]

            # comparedim_list.append(compare_dims)

            # flag2=
            if set(cur_mechs) & set(Exp.image_labels)  != set() :
                D_input_dim = 3
                D_output_dim = 1
                num_filters = Exp.IMAGE_FILTERS
                cur_discriminator = DigitImageDiscriminator(
                    image_dim=D_input_dim,
                    label_dims=compare_dims,
                    num_filters=num_filters[::-1],
                    output_dim=D_output_dim
                ).to(Exp.DEVICE)
            else:
                rep_dim=0
                if set(all_compare_Var) & set(Exp.rep_labels) != set():
                    rep_dim = 10
                cur_discriminator= ControllerDiscriminator(Exp, input_dim=compare_dims+ rep_dim).to(Exp.DEVICE)

            discriminatorsMech[hnode]=cur_discriminator
            doptimizersMech[hnode] = torch.optim.Adam(cur_discriminator.parameters(), lr=Exp.learning_rate, betas=Exp.betas, weight_decay=Exp.discriminator_decay)


    # saving all discriminators
    # # need to load discriminator for both observation and interventional dataset
    if True in load_which_models.values():
        # dfile = Exp.LOAD_MODEL_PATH + "/checkpoints_discriminator/epochLast.pth"
        dfile = disc_check_point

        if Path(dfile).is_file():
            checkpointx = torch.load(dfile, map_location="cuda")

        # for lbid, label in enumerate(Exp.label_names):
            # if load_which_models[label] == True:
            var_list= "".join(x for x in cur_mechs)
            for id, _ in enumerate(discriminatorsMech):
                if "dstate_dict"+var_list+str(id) not in checkpointx:
                    continue
                discriminatorsMech[id].load_state_dict(checkpointx["dstate_dict"+var_list+str(id)])
                doptimizersMech[id].load_state_dict(checkpointx["doptimizer" + var_list+str(id)])

                for param_group in doptimizersMech[id].param_groups:
                        param_group["lr"] = Exp.learning_rate
        else:
            print("No discriminator loaded")

    return discriminatorsMech, doptimizersMech


def get_generated_labels(Exp, label_generators, label_noises, conf_noises, intervened, chosen_labels, mini_batch, **kwargs):
    if not label_noises:
        for name in Exp.label_names:
            if name not in Exp.image_labels:
                label_noises[Exp.exogenous[name]] = torch.randn(mini_batch, Exp.NOISE_DIM).to(
                    Exp.DEVICE)  # white noise. no bias

    if not conf_noises:
        for label in Exp.label_names:
            confounders = Exp.latent_conf[label]
            for conf in confounders:  # no confounder name, only their sequence matters here.
                conf_noises[conf] = torch.randn(mini_batch, Exp.CONF_NOISE_DIM).to(Exp.DEVICE)  # white noise. no bias

    max_in_top_order = max([Exp.label_names.index(lb) for lb in chosen_labels])
    # print("max_in_top_order", max_in_top_order)
    gen_labels = {}
    for lbid, label in enumerate(Exp.Observed_DAG):
        if lbid > max_in_top_order:  # we dont need to produce the rest of the variables.
            break

        # print(lbid, label)
        # first adding exogenous noise
        Noises = []
        if label not in Exp.image_labels:
            Noises.append(label_noises[Exp.exogenous[label]])  # error here

        # secondly, adding confounding noise
        for conf in Exp.latent_conf[label]:
            Noises.append(conf_noises[conf])


        # getting observed parent values
        parent_gen_labels = []
        for parent in Exp.Observed_DAG[label]:
            parent_gen_labels.append(gen_labels[parent])

        if label in intervened.keys():
            if torch.is_tensor(intervened[label]):
                gen_labels[label] = intervened[label]
            else:
                gen_labels[label] = torch.ones(mini_batch, Exp.label_dim[label]).to(Exp.DEVICE) * 0.00001
                gen_labels[label][:, intervened[label]] = 0.99999

        elif label in Exp.image_labels:

            # if 'true_scm' in kwargs and kwargs['true_scm']==True:  #producing images from function
            #     parent_gen_labels = torch.tensor(map_dictfill_to_discrete(Exp, {par:gen_labels[par] for par in Exp.Observed_DAG[label]} , Exp.Observed_DAG[label])).to(Exp.DEVICE)
            #     gen_image= produce_uniform_images(Exp, 0, parent_gen_labels, mini_batch , True )
            #
            #     transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
            #     digit_images = [torch.unsqueeze(transform(img.astype(uint8)), dim=0).to(Exp.DEVICE) for img in gen_image]
            #     gen_labels[label]= torch.cat(digit_images, 0)
            #     continue

            Noises = []
            image_noise = torch.randn(mini_batch, Exp.IMAGE_NOISE_DIM).view(-1, Exp.IMAGE_NOISE_DIM, 1, 1).to(
                Exp.DEVICE)
            Noises.append(image_noise)
            for conf in Exp.latent_conf[label]:
                Noises.append(conf_noises[conf].view(-1, Exp.CONF_NOISE_DIM, 1, 1).to(Exp.DEVICE))

            # converting continuous fill parents to discrete fill
            # if len(parent_gen_labels)==0:   #no image parent? Just feed noise
            #     parent_gen_labels= Noises
            # else:
            parent_gen_labels = [torch.cat(parent_gen_labels, 1).unsqueeze(2).unsqueeze(3)]
            # dims_list = [Exp.label_dim[lb] for lb in Exp.Observed_DAG[label]]
            # parent_gen_labels = map_fill_to_discrete(Exp, parent_gen_labels, dims_list)
            # parent_gen_labels = [get_multiple_labels_fill(Exp, parent_gen_labels, dims_list, isImage_labels=True, image_size=1)]

            gen_labels[label] = label_generators[label](Noises, parent_gen_labels) #sending lists

        # elif set(Exp.Observed_DAG[label]) & set(Exp.rep_labels) != set():
        elif label in Exp.rep_labels:
            # gen_labels[label] = label_generators[label](Exp, parent_gen_labels, gumbel_noise=None, hard=False)
            img= parent_gen_labels[0]
            if Exp.DAG_desc=="imageMediator":   #encoder for mnist images.
                label_data= parent_gen_labels[1]  #onehot?
                par= Exp.Observed_DAG[label][1]
                dim_list= [Exp.label_dim[par]]
                gen_labels[label] = label_generators[label](Exp, img, label_data, dim_list,  isLatent=True)  #sending onehot labels?
            elif Exp.DAG_desc=="celebAtrain": #encoder for celeba images
                gen_labels[label] = label_generators[label](Exp, img, isLatent=True)  # sending onehot labels?
            # Implemented frontdoor for two graphs.

        elif set(Exp.Observed_DAG[label]) & set(Exp.image_labels) != set():
            for idx, par_label in enumerate(parent_gen_labels):
                if len(par_label.shape)<4:
                    parent_gen_labels[idx]= par_label.unsqueeze(2).unsqueeze(3).repeat(1, 1, Exp.IMAGE_SIZE, Exp.IMAGE_SIZE)

            for idx, par_noise in enumerate(Noises):
                if len(par_noise.shape)<4:
                    Noises[idx]= par_noise.unsqueeze(2).unsqueeze(3).repeat(1, 1, Exp.IMAGE_SIZE, Exp.IMAGE_SIZE)

            all_inputs = torch.cat(Noises+ parent_gen_labels, dim=1)
            gen_labels[label] = label_generators[label](Exp, all_inputs , gumbel_noise=None, hard=False)
            # print('hrere')
            # img, c= parent_gen_labels[1], parent_gen_labels[0]
            # gen_labels[label] = label_generators[label](Exp, img, c, gumbel_noise=None, hard=False)
        else:
            gn=None
            hard= False
            if "gumbel_noise" in kwargs:
                gn=kwargs["gumbel_noise"][label]
            if "hard" in kwargs:
                hard= kwargs["hard"]
            gen_labels[label] = label_generators[label](Exp, Noises, parent_gen_labels, gumbel_noise=gn, hard=hard)
            # print('here2')

    return_labels = {}
    for label in chosen_labels:
        return_labels[label] = gen_labels[label]

    return return_labels




def get_fake_distribution(Exp, label_generators, intv_key, compare_Var, sample_size ):
    generated_labels_dict = get_generated_labels(Exp, label_generators, {}, {}, dict(intv_key), compare_Var,
                                                 sample_size)
    generated_labels_full = map_dictfill_to_discrete(Exp, generated_labels_dict, compare_Var)
    fake_dist_dict = get_joint_distributions_from_samples(Exp, compare_Var, generated_labels_full)

    return fake_dist_dict