import itertools
import os

import numpy as np
import torch
# import seaborn as sns
# import matplotlib.pyplot as plt
# import collections
# old constants
# from datetime import datetime


def init_weights(m):  # for generator and discriminator, they are initialized inside the class
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def map_to_multiple_states(ara, label_dim):  # want to change it to alpha*x
    input = torch.tensor(ara)

    portion = 1.0 / label_dim

    for i in range(label_dim):
        rep_val = torch.tensor(i, dtype=input.dtype).to(Global_Constants.DEVICE)
        input = torch.where((portion * i <= input) & (input < portion * (i + 1)), rep_val, input)

    return input


def map_to_exact_discrete(ara):
    ara = torch.tensor(ara)
    ara = torch.where(ara <= 0.5, 0, 1)
    return ara



def map_dictfill_to_discrete(Exp, generated_labels_dict, compare_Var):
    y_dims = sum([Exp.label_dim[lb] for lb in compare_Var])
    ret = list(generated_labels_dict.values())
    generated_labels_full = torch.cat(ret, 1).view(-1, y_dims)
    dims_list = [Exp.label_dim[lb]for lb in compare_Var]
    generated_labels_full = map_fill_to_discrete(Exp, generated_labels_full, dims_list).detach().cpu().numpy().astype(int)

    return generated_labels_full


def map_fill_to_discrete(Exp, ara, dims_list):
    each_col = []

    start,end=0,0
    for dim in dims_list:
        end=start+dim
        indices = torch.argmax(ara[:, start: end], dim=1).view(-1,1)  # for each variable
        each_col.append(indices)
        start= end


    # for id in range(int(ara.shape[1] / Exp.label_dim)):
    #     temp = ara[:, id * Exp.label_dim: (id + 1) * Exp.label_dim]
    #     indices = torch.argmax(ara[:, id * Exp.label_dim: (id + 1) * Exp.label_dim], dim=1).view(-1,1)  # for each variable
    #     each_col.append(indices)

    result = torch.cat(each_col, 1)
    return result


def map_fill_to_discrete_max_val(Exp, ara):
    each_col = []
    for id in range(int(ara.shape[1] / Exp.label_dim)):
        max_val, indices = torch.max(ara[:, id * Exp.label_dim: (id + 1) * Exp.label_dim], dim=1)  # for each variable
        each_col.append(max_val.view(-1, 1))

    result = torch.cat(each_col, 1)
    return result


def get_multiple_labels_fill(Exp, data_input, dims_list, isImage_labels, **kwargs):  # dist_conds is a list of conditions for each label

    labels_fill = []
    for id in range(data_input.shape[1]):

        label_dim = dims_list[id]
        if isImage_labels:
            fill = torch.zeros([label_dim, label_dim, kwargs["image_size"], kwargs["image_size"]]).to(Exp.DEVICE)
        else:
            fill = torch.zeros([label_dim, label_dim]).to(Exp.DEVICE)
        # for i in range(label_dim):
        #     for j in range(label_dim):
        #         # fill[i,j]=0.00001
        #         fill[i, j] = 0

        for i in range(label_dim):
            # fill[i, i] = 0.99999
            if isImage_labels:
                fill[i, i, :, :] = 1
            else:
                fill[i, i] = 1

        current_label = data_input[:, id].type(torch.LongTensor).view(-1, 1).to(Exp.DEVICE)
        filled_real_label = fill[current_label].to(Exp.DEVICE)
        if isImage_labels:
            ret = filled_real_label.view(-1, label_dim, kwargs["image_size"], kwargs["image_size"])
        else:
            ret = filled_real_label.view(-1, label_dim)

        labels_fill.append(ret)
    real_labels_fill = torch.cat(labels_fill, 1).to(Exp.DEVICE)  # this one

    return real_labels_fill




def fill2d_to_fill4d(Exp, data_input, **kwargs):  # dist_conds is a list of conditions for each label

    dim1= data_input.shape[0]
    dim2= data_input.shape[1]
    new_data_input = torch.zeros([ dim1, dim2 , kwargs["image_size"], kwargs["image_size"]]).to(Exp.DEVICE)

    for i in range(dim1):
        for j in range(dim2):
            new_data_input[i, j, :, :] = data_input[i, j]

    return new_data_input




def get_label_fill(label_dim):
    fill = torch.zeros([label_dim, label_dim])  # A label_dim x label_dim identity matrix

    for i in range(label_dim):
        for j in range(label_dim):
            # fill[i,j]=0.00001
            fill[i, j] = 0

    for i in range(label_dim):
        # fill[i, i] = 0.99999
        fill[i, i] = 1

    return fill


def get_label_onehot(label_dim):
    onehot = torch.zeros(label_dim, label_dim)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(label_dim, 1), 1).view(label_dim, label_dim)

    return onehot


def generate_permutations(dim_list):
    sequences=[]
    for dim in dim_list:
        sequences.append([i for i in range(dim)])

    lst = []
    for p in itertools.product(*sequences):
        lst.append(p)

    np_ara = np.array(lst)
    return np_ara


# https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

