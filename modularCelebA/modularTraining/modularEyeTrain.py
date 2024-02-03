# CelebA image generation using Conditional DCGAN
import os
import pickle
import sys

import pandas as pd
import torch
from tqdm import tqdm

# from Image_Mediator_Evaluation import  plot_image_ara
from ModularUtils import ControllerConstants
from ModularUtils.ControllerConstants import get_multiple_labels_fill, map_dictfill_to_discrete
from ModularUtils.Discriminators import ControllerDiscriminator
from ModularUtils.FrontBackDoorCalculation import estiamte_ate_backdoor_direct
from ModularUtils.FunctionsDistribution import get_joint_distributions_from_samples, calculate_TVD, calculate_KL
from ModularUtils.Generators import ControllerGenerator
from ModularUtils.FunctionsConstant import load_label_dataset, asKey, initialize_results, load_image_dataset, getdoKey, \
    get_dataset
from ModularUtils.Experiment_Class import Experiment
from ModularUtils.FunctionsTraining import get_training_variables, labels_image_gradient_penalty, calc_gradient_penalty, \
    save_checkpoint, image_gradient_penalty, save_results
from torch import optim as optim

from modularTraining.TrainEyeglassClassifiers.celeba_eye_graph import set_celeba_eye
import modularTraining.constant_paths as const


def get_fake_distribution(Exp, label_generators, intv_key, compare_Var, sample_size ):
    generated_labels_dict = get_generated_labels(Exp, label_generators, dict(intv_key), compare_Var,
                                                 sample_size)
    generated_labels_full = map_dictfill_to_discrete(Exp, generated_labels_dict, compare_Var)
    fake_dist_dict = get_joint_distributions_from_samples(Exp, compare_Var, generated_labels_full)

    return fake_dist_dict


def get_discriminators(Exp):
    critic={}
    optimizer={}

    critic['H1'] = ControllerDiscriminator(Exp, input_dim=2+2).to(Exp.DEVICE)

    for key in critic:
        optimizer[key]= torch.optim.Adam(critic[key].parameters(), lr=Exp.learning_rate, betas=Exp.betas,
                         weight_decay=Exp.discriminator_decay)

    return critic, optimizer

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
            # load stylegan
            pass


        else:
            label_generators[label] = ControllerGenerator(Exp, input_dim=noise_dims + parent_dims,output_dim=Exp.label_dim[label]).to(Exp.DEVICE)
            label_generators[label].apply(ControllerConstants.init_weights)
            optimizersMech[label] = torch.optim.Adam(label_generators[label].parameters(), lr=Exp.learning_rate,
                                                     betas=Exp.betas,  weight_decay=Exp.generator_decay)


    return label_generators, optimizersMech


def get_generated_labels(Exp, label_generators, intervened, chosen_labels, mini_batch, **kwargs):

    label_noises={}
    for name in Exp.label_names:
        if name not in Exp.image_labels:
            label_noises[Exp.exogenous[name]] = torch.randn(mini_batch, Exp.NOISE_DIM).to(
                Exp.DEVICE)  # white noise. no bias

    conf_noises={}
    for label in Exp.label_names:
        confounders = Exp.latent_conf[label]
        for conf in confounders:  # no confounder name, only their sequence matters here.
            conf_noises[conf] = torch.randn(mini_batch, Exp.CONF_NOISE_DIM).to(Exp.DEVICE)  # white noise. no bias

    max_in_top_order = max([Exp.label_names.index(lb) for lb in chosen_labels])
    gen_labels = {}
    for lbid, label in enumerate(Exp.Observed_DAG):
        if lbid > max_in_top_order:  # we dont need to produce the rest of the variables.
            break

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

        else:
            gen_labels[label] = label_generators[label](Exp, Noises, parent_gen_labels, gumbel_noise=None, hard=False)

    return_labels = {}
    for label in chosen_labels:
        return_labels[label] = gen_labels[label]

    return return_labels



def train_H1(Exp, hn, cur_mechs, label_generators, G_optimizers, critic, D_optimizer,
                           label_data):


    current_real_label = torch.cat(list(label_data.values()), dim=1)
    mini_batch = current_real_label.shape[0]
    dims_list = [2,2]
    real_labels_fill = get_multiple_labels_fill(Exp, current_real_label, dims_list, isImage_labels=False)

    generated_labels_dict = get_generated_labels(Exp, label_generators, {}, ['Male', 'Eyeglasses'], mini_batch)
    ret = list(generated_labels_dict.values())
    generated_labels_fill = torch.cat(ret, 1)

    D_losses = []
    for crit_ in range(Exp.CRITIC_ITERATIONS):
        D_real_decision_obs = critic[hn](real_labels_fill).squeeze()
        D_fake_decision_obs = critic[hn](generated_labels_fill).squeeze()
        gp_obs = calc_gradient_penalty(critic[hn], real_labels_fill, generated_labels_fill, device=Exp.DEVICE)

        D_loss_obs = (-  (torch.mean(D_real_decision_obs) - torch.mean(D_fake_decision_obs)) + Exp.LAMBDA_GP * gp_obs)
        D_losses.append((D_loss_obs).data)
        critic[hn].zero_grad()
        D_loss_obs.backward(retain_graph=True)
        D_optimizer[hn].step()

    #%%%%%%%%%%%%%%%%%%% generator  training  %%%%%%%%%%%%%%%%%%%
    # Back propagation
    for mech in cur_mechs:
        label_generators[mech].zero_grad()

    D_fake_decision_obs = critic[hn](generated_labels_fill).squeeze()
    G_loss = -torch.mean(D_fake_decision_obs)

    # Back propagation
    G_loss.backward()

    for mech in cur_mechs:
        G_optimizers[mech].step()

    D_loss = torch.mean(torch.FloatTensor(D_losses))  # just mean of losses

    return G_loss.data, D_loss.data




def labelMain(Exp, cur_hnodes, label_generators, G_optimizers, discriminators, D_optimizers, label_data, image_data,
              tvd_diff, kl_diff):

    medD_loader = torch.utils.data.DataLoader(dataset=label_data['Male'], batch_size=Exp.batch_size, shuffle=False, drop_last=True)
    medC_loader = torch.utils.data.DataLoader(dataset=label_data['Eyeglasses'], batch_size=Exp.batch_size, shuffle=False, drop_last=True)

    label_batch = []
    for medD_batch, medC_batch in zip(medD_loader, medC_loader):
        label_batch.append({'Male': medD_batch.view(-1, 1), 'Eyeglasses': medC_batch.view(-1, 1)})


    iteration = 0
    num_batches = len(label_batch)


    for batchno in range(num_batches):
        labels = label_batch[batchno]

        # for hn in cur_hnodes:

        hn = "H1"
        cur_mechs= cur_hnodes[hn]
        g_loss, d_loss = train_H1(Exp, hn, cur_mechs, label_generators, G_optimizers, discriminators,
                                                  D_optimizers, labels)

        print('Epoch [%d/%d], Step [%d/%d],' % (
            Exp.curr_epoochs + 1, Exp.num_epochs, iteration + 1, num_batches),
          'mechanism: ',cur_hnodes[hn],  ' D_loss: %.4f, G_loss: %.4f' % (d_loss.data, g_loss.data))


        # Annealing
        tot_iter = Exp.curr_epoochs * num_batches + iteration
        if tot_iter % 100 == 0:
            Exp.anneal_temperature(tot_iter)

        Exp.D_avg_losses.append(torch.mean(d_loss))
        Exp.G_avg_losses.append(torch.mean(g_loss))
        iteration += 1
    #


    if (Exp.curr_epoochs + 1) % 5 == 0:

        fake_dist_dict = get_fake_distribution(Exp, label_generators, {}, Exp.label_names, sample_size=20000)
        current_real_label= torch.cat(list(label_data.values()), dim=1)
        dataset_dist_dict = get_joint_distributions_from_samples(Exp, Exp.label_names,
                                                                 current_real_label.detach().cpu().numpy().astype(
                                                                     int))
        obs_tvd = calculate_TVD(fake_dist_dict, dataset_dist_dict, doPrint=False)
        obs_kl = calculate_KL(fake_dist_dict, dataset_dist_dict, doPrint=False)
        tvd_diff['joint'].append(round(obs_tvd, 4))  # todo: fix it
        kl_diff['joint'].append(round(obs_kl, 4))

        ll = -min(10, len(list(tvd_diff.values())[0]))
        for dist in tvd_diff:
            print("###", dist, " loss%:", [round(val, 4) for val in tvd_diff[dist][ll:]])

    #
    if (Exp.curr_epoochs + 1) % 100 == 0:
        save_checkpoint(Exp, Exp.SAVED_PATH, label_generators, G_optimizers, discriminators,  D_optimizers)
        print(Exp.curr_epoochs, ":model saved at ", Exp.SAVED_PATH)
    return




if __name__ == "__main__":


    args = sys.argv
    if len(args) == 1:
        exp_name = 'celeba-eye-train'
    else:
        exp_name = args[1]


    Exp = Experiment(exp_name, set_celeba_eye,
                     Temperature=1,
                     temp_min=0.1,
                     G_hid_dims=[256, 256],
                     D_hid_dims=[256, 256],
                     # IMAGE_FILTERS=[512, 256, 128],
                     IMAGE_FILTERS=[128, 64, 32],
                     CRITIC_ITERATIONS=1,
                     LAMBDA_GP=10,
                     learning_rate= 1e-4,
                     Synthetic_Sample_Size=40000,
                     batch_size=200,
                     noise_states=64,
                     latent_state=4,
                     ENCODED_DIM=10,
                     Data_intervs=[{}],
                     num_epochs=301,
                     NOISE_DIM =128,
                     CONF_NOISE_DIM=128,
                     new_experiment=True
                     )


    print(Exp.Data_intervs)
    Exp.intv_batch_size = Exp.batch_size
    os.makedirs(Exp.SAVED_PATH, exist_ok=True)

    Exp.load_which_models = {"Male": False, "Eyeglasses": False}
    cur_hnodes = {"H1":["Male", "Eyeglasses"]}


    Exp.LAMBDA_GP=10
    label_generators, optimizersMech = get_generators(Exp, Exp.load_which_models, None)
    discriminatorsMech, doptimizersMech = get_discriminators(Exp)  #



    # image_size = 64
    # file = f"../images{image_size}/8_attribute_40k_celeba_train.pkl"
    # with open(file, 'rb') as f:
    #     real_data = pickle.load(f)


    exp_name = "sex_eyeglass"
    dom_name = "domain1"
    targetAtt = 'Eyeglasses'
    lb_file = f'/{const.project_root}/modularCelebA/{exp_name}/8_attribute_celeba_{dom_name}.pkl'
    with open(lb_file, 'rb') as f:
        domain_dataset = pickle.load(f)

    domain1_labels= domain_dataset

    label_data={}
    label_data['Male'] = torch.tensor(domain1_labels['Male']).view(-1,1).to(Exp.DEVICE)
    label_data['Eyeglasses'] = torch.tensor(domain1_labels['Eyeglasses']).view(-1,1).to(Exp.DEVICE)

    image_data= {}

    current_real_label= torch.cat([label_data['Male'], label_data['Eyeglasses']], dim=1)
    dataset_dist_dict = get_joint_distributions_from_samples(Exp, Exp.label_names,
                                                             current_real_label.detach().cpu().numpy().astype(
                                                                 int))



    tvd_diff, kl_diff = {'joint':[]}, {'joint':[]}
    mech_tvd = 0
    print("Starting training new mechanism")

    for epoch in range(Exp.num_epochs):
        Exp.curr_epoochs = epoch
        labelMain(Exp, cur_hnodes, label_generators, optimizersMech, discriminatorsMech, doptimizersMech, label_data,
                  image_data, tvd_diff, kl_diff)


