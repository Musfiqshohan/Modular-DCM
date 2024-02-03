import copy
import os
import pickle

# import numpy as np
import torch
# import torch.nn.functional as F
import torchvision
# import torch.nn as nn
# import networkx as nx
# import matplotlib.pyplot as plt
# from graphviz import Digraph

import graphviz

# from graphviz import Source
#


def get_training_variables(Exp, cur_mechs, interv_no, intv_key):
    all_compare_Var = []
    for mech in cur_mechs:
        # if interv_no not in Exp.train_mech_dict[mech]:
        #     continue
        ret = [lb for lb in Exp.train_mech_dict[mech][interv_no]["compare"] if lb not in all_compare_Var]
        all_compare_Var += ret

    # parents of all joint variables are suitable for intervention.
    intervened_Var = list(intv_key.keys())
    for mech in all_compare_Var:
        intervened_Var += [lb for lb in Exp.train_mech_dict[mech][interv_no]["parents"] if
                    lb not in intervened_Var + all_compare_Var+Exp.image_labels]

    compare_Var =[] # [lb for lb in all_compare_Var ]
    for lb in all_compare_Var:
        # if lb in Exp.image_labels:
        #     compare_Var.append("R"+lb)
        # else:
        if lb not in Exp.image_labels+Exp.rep_labels:
            compare_Var.append(lb)

    real_data_vars=[]
    for lb in intervened_Var+ compare_Var:
        if lb not in real_data_vars:
            real_data_vars+=[lb]

    return all_compare_Var, compare_Var, intervened_Var, real_data_vars
    #HnUAn+ U images, HnUAn+\{images,RI}, pa(HnUAn+\RI),  HnUAn+ U pa(HnUAn+)
    #all without any RI.



def getdoKey(obs_Var, intv_key):
    query_str = "P(" + ",".join(x for x in obs_Var)

    if len(intv_key) != 0:
        if type(intv_key) == dict:
            query_str = query_str + "|do(" + ",".join(x for x in intv_key.keys()) + "_" + "".join(str(x) for x in intv_key.values())+")"
        else:
            query_str = query_str + "|do(" + "".join(x for x in intv_key)+")"

    # if len(intv_key) == 0:
    #     query_str += "[]"
    query_str += ")"
    return query_str


def draw_true_graph(DAG):
    f = graphviz.Digraph(filename="output/plainorganogram1.gv")
    K = {}
    for id, label in enumerate(DAG.keys()):
        K[label] = str(id)
        f.node(K[label], label)

    for label in DAG:
        for par in DAG[label]:
            f.edge(K[par], K[label])
    print(f.source)
    f.render(view=True)




def asKey(a):
    return tuple(sorted(a.items()))

def build_compares(confTochild, Observed_DAG, label_names, intervened):
    # confTochild, Observed_DAG, label_names = confTochild, Observed_DAG, label_names
    confTochild = copy.deepcopy(confTochild)
    Observed_DAG = copy.deepcopy(Observed_DAG)

    for U in list(confTochild.keys()):
        if list(set(confTochild[U]) & set(intervened)):
            del confTochild[U]  # since no latent confounders to intervened vars
        # confTochild[U]=[label for label in confTochild[U] if label not in intervened]   #since no latent confounders to intervened vars

    for label in intervened:
        Observed_DAG[label]=[]

    lst = list(confTochild.keys())

    added = {}
    compare = {}
    for i in range(len(lst)):
        Ui = lst[i]
        if Ui in added:
            continue

        print(Ui)
        added[Ui] = 1
        children = set(confTochild[Ui])

        for j in range(i + 1, len(lst)):
            Uj = lst[j]
            newchild = set(confTochild[Uj])
            joint = children | newchild
            if len(joint) < len(children) + len(newchild):
                children = children | newchild
                added[Uj] = 1

        c_comp = list(sorted(children, key=lambda d: label_names.index(d)))
        print(c_comp)
        for lb, label in enumerate(c_comp):
            cur = [label] + Observed_DAG[label]
            if lb - 1 >= 0:
                cur += compare[c_comp[lb - 1]]
            compare[label] = list(sorted(set(cur), key=lambda d: label_names.index(d)))
            # print(compare)

    for label in label_names:
        all= set([label] +Observed_DAG[label])
        if label in compare:
            all = all | set(compare[label])

        compare[label] = list(sorted(all, key=lambda d: label_names.index(d)))

    return compare


def top_sort_dict(dict, top_list):
    # print("getting latent confounders and observed variables.")
    new_dict = {}
    for nd in top_list:
        if nd in dict.keys():
            new_dict[nd] = dict[nd]

    return new_dict


def top_sort_list(lst, top_list):
    # print("getting latent confounders and observed variables.")
    new_lst = []
    for nd in top_list:
        if nd in lst:
            new_lst.append(nd)

    return new_lst





def save_results(Exp, saved_path, all_generated_labels, all_real_labels,  tvd_diff, kl_diff, G_avg_losses, D_avg_losses):


    for dist in tvd_diff:
        os.makedirs(saved_path + "/tvd", exist_ok=True)
        os.makedirs(saved_path + "/kl", exist_ok=True)
        torch.save(torch.FloatTensor(tvd_diff[dist]), saved_path + "/tvd/"+dist)
        torch.save(torch.FloatTensor(kl_diff[dist]), saved_path + "/kl/"+dist)


    for key in all_generated_labels:
        torch.save(all_generated_labels[key], saved_path + f"/{str(key)}generated_labels")
        torch.save(torch.FloatTensor(G_avg_losses), saved_path + f"/{str(key)}G_avg_losses")

        if key in all_real_labels:
            torch.save(all_real_labels[key], saved_path + f"/{str(key)}real_labels")
        torch.save(torch.FloatTensor(D_avg_losses), saved_path + f"/{str(key)}D_avg_losses")



def save_checkpoint(Exp , saved_path, cur_mechs, label_generators, G_optimizers, label_discriminators, D_optimizers):
    if Exp.SAVE_MODEL:

        print("=> Saving checkpoint")
        # gen_checkpoint = {"epoch":Exp.curr_epoochs, "trained":[lb for lb, isLoad in Exp.load_which_models.items() if isLoad==True]+[cur_mech]}
        gen_checkpoint = {"epoch":Exp.curr_epoochs, "trained":[lb for lb, isLoad in Exp.load_which_models.items() if isLoad==True]}
        for label in label_generators:
            gen_checkpoint[label + "state_dict"] = label_generators[label].state_dict()
            gen_checkpoint["optimizer" + label]= G_optimizers[label].state_dict()

        # Exp.checkpoints["generator"].append(gen_checkpoint)

        os.makedirs(saved_path + f"/checkpoints_generators", exist_ok=True)
        gfile = saved_path + f"/checkpoints_generators/epoch{Exp.curr_epoochs:03}.pth"
        last_gfile = saved_path + f"/checkpoints_generators/epochLast.pth"
        torch.save(gen_checkpoint, gfile)
        torch.save(gen_checkpoint, last_gfile)



        #--------

        # disc_checkpoint = {"epoch":Exp.curr_epoochs, "trained":[lb for lb, isLoad in Exp.load_which_models.items() if isLoad==True]+[cur_mech]}
        disc_checkpoint = {"epoch":Exp.curr_epoochs, "trained":[lb for lb, isLoad in Exp.load_which_models.items() if isLoad==True]}

        for intvno in range(len(Exp.Data_intervs)):
            for obsno in range(len(Exp.Data_observs)):
        # for label in label_discriminators:
        #     for id in range (len(label_discriminators[label])):
                disc_checkpoint["dstate_dict"+str(intvno)+str(obsno)] = label_discriminators[intvno][obsno].state_dict()
                disc_checkpoint["doptimizer" + str(intvno)+str(obsno)]= D_optimizers[intvno][obsno].state_dict()



        # Exp.checkpoints["discriminator"].append(disc_checkpoint)
        os.makedirs(saved_path + f"/checkpoints_discriminator", exist_ok=True)
        dfile = saved_path + f"/checkpoints_discriminator/epoch{Exp.curr_epoochs:03}.pth"
        last_dfile = saved_path + f"/checkpoints_discriminator/epochLast.pth"
        torch.save(disc_checkpoint, dfile)
        torch.save(disc_checkpoint, last_dfile)



def load_checkpointed_generators(saved_path, label_generators, G_optimizers, lr):
    gfile = saved_path + "/checkpoints_generators.pth"
    checkpoint = torch.load(gfile, map_location="cuda")

    for label in label_generators:
        label_generators[label].load_state_dict(checkpoint[label + "state_dict"])

    # print("generator Load", checkpoint)

    for i, label in enumerate(G_optimizers):
        G_optimizers[label].load_state_dict(checkpoint["optimizer" + str(i)])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in G_optimizers[label].param_groups:
            param_group["lr"] = lr


def load_checkpointed_discriminators(saved_path, label_discriminators, D_optimizers, lr):
    dfile = saved_path + "/checkpoints_discriminator.pth"
    checkpoint = torch.load(dfile, map_location="cuda")

    for id in range(len(label_discriminators)):
        label_discriminators[id].load_state_dict(checkpoint["dstate_dict" + str(id)])

    for id, optimizer in enumerate(D_optimizers):
        optimizer.load_state_dict(checkpoint["doptimizer" + str(id)])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

# load_checkpoint(Exp.SAVED_PATH, label_generators, [G_optimizers], discriminators, D_optimizers, Exp.learning_rate)
def load_checkpoint(saved_path, label_generators, G_optimizers, label_discriminators, D_optimizers, lr):
    print("=> Loading checkpoint")

    load_checkpointed_generators(saved_path, label_generators, G_optimizers, lr)
    load_checkpointed_discriminators(saved_path, label_discriminators, D_optimizers, lr)



def plot_to_tensorboard(
        writer, loss_critic, loss_gen, real, fake, tensorboard_step
):
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)

    with torch.no_grad():
        # take out (up to) 8 examples to plot
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)



# compare two nn model
def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    print("validating state dicts")
    if len(model_state_dict_1) != len(model_state_dict_2):
        print(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1:]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1:]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
            model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            print(f"Tensor mismatch: {v_1} vs {v_2}")
            return False

    print("No mismatch")


def get_dataset(Exp, label, dno):

    dataset = []
    # for feature in ["feature"]:
        # file_name = Exp.file_roots[dno] + label + feature + ".pkl"
    file_name = Exp.file_roots + "intv" + str(dno) + label + ".pkl"
    # file_name = Exp.file_roots + "interv" + str(dno) + label + ".pkl"
    with open(file_name, 'rb') as fp:
        label_data = pickle.load(fp)
    label_data = torch.FloatTensor(label_data)
    label_size = len(label_data)
    dataset.append(label_data.view(label_size, 1))

    result_dataset = torch.cat(dataset, 1).to(Exp.DEVICE)
    print(result_dataset.shape)
    return result_dataset