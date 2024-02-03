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
            fill = torch.zeros([label_dim, label_dim, kwargs["more_dimsize"], kwargs["more_dimsize"]]).to(Exp.DEVICE)
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
            ret = filled_real_label.view(-1, label_dim, kwargs["more_dimsize"], kwargs["more_dimsize"])
        else:
            ret = filled_real_label.view(-1, label_dim)

        labels_fill.append(ret)
    real_labels_fill = torch.cat(labels_fill, 1).to(Exp.DEVICE)  # this one

    return real_labels_fill




def fill2d_to_fill4d(Exp, data_input, **kwargs):  # dist_conds is a list of conditions for each label

    dim1= data_input.shape[0]
    dim2= data_input.shape[1]
    new_data_input = torch.zeros([ dim1, dim2 , kwargs["more_dimsize"], kwargs["more_dimsize"]]).to(Exp.DEVICE)

    for i in range(dim1):
        for j in range(dim2):
            new_data_input[i, j, :, :] = data_input[i, j]



    # labels_fill = []
    #
    # start,end=0,0
    # for dim in dims_list:
    #     end=start+dim
    #     data= data_input[:, start: end]
    #     label_dim = dim
    #     fill = torch.zeros([label_dim, label_dim, kwargs["more_dimsize"], kwargs["more_dimsize"]])
    #
    #     for i in range(label_dim):
    #         fill[i, i, :, :] = data[i,i]
    #
    #     ret = fill.view(-1, label_dim, kwargs["more_dimsize"], kwargs["more_dimsize"])
    #     labels_fill.append(ret)
    #
    # real_labels_fill = torch.cat(labels_fill, 1).to(Exp.DEVICE)  # this one

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


def set_frontdoor_wo_latents():
    global num_labels, label_names, complete_labels, Observed_DAG, Complete_DAG, intervened_var, observed_var, noise_dist, DAG_desc, Complete_DAG_desc, exogenous, latent_conf, joint_dist, SAVED_PATH
    DAG_desc = "frontdoor-wo-latents"
    SAVED_PATH = SAVED_PATH + DAG_desc

    Complete_DAG_desc = "frontdoor"
    Complete_DAG["nU_xy"] = []
    Complete_DAG["X"] = ["nU_xy"]
    Complete_DAG["Z"] = ["X"]
    Complete_DAG["Y"] = ["nU_xy", "Z"]
    complete_labels = list(Complete_DAG.keys())

    Observed_DAG["nU_xy"] = []
    Observed_DAG["X"] = ["nU_xy"]
    Observed_DAG["Z"] = ["X"]
    Observed_DAG["Y"] = ["nU_xy", "Z"]
    intervened_var = ["X"]
    observed_var = ["Y"]

    label_names = list(Observed_DAG.keys())
    num_labels = len(label_names)

    # need to follow the topological order
    # latent_conf = {"nU_xy":[], "X": [], "Z": [], "Y": []}
    # confTochild =
    exogenous = {"nU_xy": "nnU_xy", "X": "nX", "Z": "nZ", "Y": "nY"}
    # exogenous = dict(sorted(exogenous.items()))

    # set with fixed seed
    noise_dist = {'nU_xy': [0.38, 0.62]}
    # noise_dist = {'nU_xy': [0.38, 0.62], 'nX': [0.77, 0.23], 'nZ': [0.44, 0.56], 'nY': [0.29, 0.71]}


def set_nonID_H_graph():
    global num_labels, label_names, complete_labels, Observed_DAG, Complete_DAG, intervened_var, observed_var, noise_dist, DAG_desc, Complete_DAG_desc, exogenous, latent_conf, joint_dist, SAVED_PATH
    DAG_desc = "non-id_H_noexos"
    SAVED_PATH = SAVED_PATH + DAG_desc

    Complete_DAG_desc = "non-id_H"
    Complete_DAG["nU_zy"] = []
    Complete_DAG["nU_zw"] = []
    Complete_DAG["nU_zx"] = []
    Complete_DAG["nU_xy"] = []
    Complete_DAG["Z"] = ["nU_zy", "nU_zw", "nU_zx"]
    Complete_DAG["X"] = ["nU_xy", "nU_zx", "Z"]
    Complete_DAG["W"] = ["nU_zw", "X"]
    Complete_DAG["Y"] = ["nU_zy", "nU_xy", "W"]
    complete_labels = list(Complete_DAG.keys())

    Observed_DAG["Z"] = []
    Observed_DAG["X"] = ["Z"]
    Observed_DAG["W"] = ["X"]
    Observed_DAG["Y"] = ["W"]
    intervened_var = ["X"]
    observed_var = ["Y"]

    label_names = list(Observed_DAG.keys())
    num_labels = len(label_names)

    # latent_conf = {"Z": ["nU_zy","nU_zw","nU_zx"], "X": ["nU_xy", "nU_zx"], "W": ["nU_zw"], "Y": ["nU_zy", "nU_xy"]}
    # confTochild=
    exogenous = {"Z": "nZ", "X": "nX", "W": "nW", "Y": "nY"}
    # exogenous = dict(sorted(exogenous.items()))

    # noise_dist = {'nU_xy': [0.38, 0.62], 'nX': [0.77, 0.23], 'nY': [0.29, 0.71]}
    noise_dist = {'nU_zy': [0.22, 0.78], 'nU_zw': [0.38, 0.62], 'nU_zx': [0.78, 0.22], 'nU_xy': [0.62, 0.38]}


def set_backdoor_wi_latents():
    global num_labels, label_names, complete_labels, Observed_DAG, Complete_DAG, intervened_var, observed_var, noise_dist, DAG_desc, Complete_DAG_desc, exogenous, latent_conf, joint_dist, SAVED_PATH
    DAG_desc = "backdoor-latents"
    SAVED_PATH = SAVED_PATH + DAG_desc

    Complete_DAG_desc = "backdoor"
    Complete_DAG["nU_zy"] = []
    Complete_DAG["Z"] = ["nU_zy"]
    Complete_DAG["X"] = ["Z"]
    Complete_DAG["Y"] = ["nU_zy", "X"]
    complete_labels = list(Complete_DAG.keys())

    Observed_DAG["Z"] = []
    Observed_DAG["X"] = ["Z"]
    Observed_DAG["Y"] = ["X"]
    intervened_var = ["X"]
    observed_var = ["Y"]

    label_names = list(Observed_DAG.keys())
    num_labels = len(label_names)

    # need to follow the topological order
    # latent_conf = {"Z": ["nU_zy"], "X": [], "Y": ["nU_zy"]}
    # confTochild=
    # latent_conf = dict(sorted(latent_conf.items()))
    exogenous = {"Z": "nZ", "X": "nX", "Y": "nY"}

    # set with fixed seed
    noise_dist = {'nU_zy': [0.38, 0.62]}


def set_ETT_Id_graph(noise_states, latent_state):
    DAG_desc = "ETT_Id"

    Complete_DAG_desc = "ETT_Id"
    Complete_DAG = {}
    Complete_DAG["L"] = []
    Complete_DAG["X1"] = []
    Complete_DAG["X2"] = ["X1", "L"]
    Complete_DAG["W"] = ["X1", "X2"]
    Complete_DAG["Y"] = ["W", "L"]
    complete_labels = list(Complete_DAG.keys())

    Observed_DAG = {}
    Observed_DAG["X1"] = []
    Observed_DAG["X2"] = ["X1"]
    Observed_DAG["W"] = ["X1", "X2"]
    Observed_DAG["Y"] = ["W"]
    label_names = list(Observed_DAG.keys())

    interv_queries = [
        # {"obs": ["X2"], "interv": ["X1"], "expr": "P(X2|do(X1))"},
        {"obs": ["W"], "interv": ["X1", "X2"], "expr": "P(W|do(X1,X2))"},
        # {"obs": ["Y"], "interv": ["X1", "X2"], "expr": "P(Y|do(X1,X2))"},
        {"obs": ["X2", "Y"], "interv": ["X1", "W"], "expr": "P(X2,Y)|do(X1,W)"},
        # {"obs": ["W", "Y"], "interv": ["X1", "X2"], "expr": "P(W,Y)|do(X1,X2)"}
    ]

    latent_conf = {"X1": [], "X2": ["L"], "W": [], "Y": ["L"]}
    confTochild = {"L": ["X2", "Y"]}
    exogenous = {"X1": "nX1", "X2": "nX2", "W": "nW", "Y": "nY"}

    cf_intervene = {"X1": 0, "X2": 0}  # check if fractions do any better
    cf_observe = ["Y"]
    cf_evidence = {"X1p": 1, "X2p": 1}

    cflabel_names = ["X1", "X1p", "X2", "X2p", "W", "Y", "L"]
    twin_map = {"X1p": "X1", "X1": "X1p", "X2p": "X2", "X2": "X2p"}

    Twin_Network = {}
    Twin_Network["X1"] = []
    Twin_Network["X2"] = []
    Twin_Network["L"] = []
    Twin_Network["X1p"] = []
    Twin_Network["X2p"] = ["X1p", "L"]
    Twin_Network["W"] = ["X1", "X2"]
    Twin_Network["Y"] = ["W", "L"]
    cf_exogenous = {"X1p": "nX1", "X2p": "nX2", "W": "nW", "Y": "nY"}

    noise_params = {"nX1": (0.1, noise_states),
                    "nX2": (0.1, noise_states),
                    "nW": (0.1, noise_states),
                    "nY": (0.1, noise_states),
                    "L": (1, latent_state)}
    # noise_dist = {}
    # np.random.seed(1)
    # noise_dist["nX1"] = np.random.dirichlet(0.1 * np.ones(noise_states), size=1)[0].tolist()
    # np.random.seed(2)
    # noise_dist["nX2"] = np.random.dirichlet(0.1 * np.ones(noise_states), size=1)[0].tolist()
    # np.random.seed(3)
    # noise_dist["nW"] = np.random.dirichlet(0.1 * np.ones(noise_states), size=1)[0].tolist()
    # np.random.seed(4)
    # noise_dist["nY"] = np.random.dirichlet(0.1 * np.ones(noise_states), size=1)[0].tolist()
    #
    # np.random.seed(0)
    # noise_dist["L"] = np.random.dirichlet(np.ones(latent_state), size=1)[0].tolist()

    return DAG_desc, Complete_DAG_desc, Complete_DAG, complete_labels, Observed_DAG, label_names, interv_queries, latent_conf, \
           confTochild, exogenous, cf_intervene, cf_observe, cf_evidence, cflabel_names, twin_map, Twin_Network, cf_exogenous, noise_params


def set_ETT_Id_graph_nolatents():
    global num_labels, label_names, complete_labels, Observed_DAG, Complete_DAG, intervened_var, observed_var, noise_dist, DAG_desc, Complete_DAG_desc, exogenous, latent_conf, SAVED_PATH, confTochild
    global cf_intervene, cf_observe, cf_evidence, cf_samples, get_3dists, interv_queries, cflabel_names, cf_exogenous, Twin_Network, twin_map
    DAG_desc = "ETT_Id_graph_nolatents"
    SAVED_PATH = SAVED_PATH + DAG_desc

    Complete_DAG_desc = "ETT_Id_graph_nolatents"
    Complete_DAG["X1"] = []
    Complete_DAG["X2"] = ["X1"]
    Complete_DAG["W"] = ["X1", "X2"]
    Complete_DAG["Y"] = ["W"]
    complete_labels = list(Complete_DAG.keys())

    Observed_DAG["X1"] = []
    Observed_DAG["X2"] = ["X1"]
    Observed_DAG["W"] = ["X1", "X2"]
    Observed_DAG["Y"] = ["W"]
    label_names = list(Observed_DAG.keys())
    num_labels = len(label_names)

    interv_queries = [
        {"obs": ["X2"], "interv": ["X1"], "expr": "P(X2|do(X1))"},
        {"obs": ["W"], "interv": ["X1", "X2"], "expr": "P(W|do(X1,X2))"},  # probelem here
        {"obs": ["Y"], "interv": ["X1", "X2"], "expr": "P(Y|do(X1,X2))"},
        {"obs": ["X2", "Y"], "interv": ["X1", "W"], "expr": "P(X2,Y)|do(X1,W)"},
        {"obs": ["W", "Y"], "interv": ["X1", "X2"], "expr": "P(W,Y)|do(X1,X2)"}
    ]

    latent_conf = {"X1": [], "X2": [], "W": [], "Y": []}
    confTochild = {}
    exogenous = {"X1": "nX1", "X2": "nX2", "W": "nW", "Y": "nY"}

    cf_intervene = {"X1": 0, "X2": 0}  # check if fractions do any better
    cf_observe = ["Y"]
    cf_evidence = {"Wp": 1}
    cf_samples = Synthetic_Sample_Size

    # get_3dists = check_ETT_ID
    cflabel_names = ["X1", "X1p", "X2", "X2p", "W", "Wp", "Y"]
    twin_map = {"X1p": "X1", "X2p": "X2", "Wp": "W"}
    Twin_Network = {}
    Twin_Network["X1"] = []
    Twin_Network["X2"] = []
    Twin_Network["X1p"] = []
    Twin_Network["X2p"] = ["X1p"]
    Twin_Network["W"] = ["X1", "X2"]
    Twin_Network["Wp"] = ["X1p", "X2p"]
    Twin_Network["Y"] = ["W"]
    cf_exogenous = {"X1p": "nX1", "X2p": "nX2", "W": "nW", "Wp": "nW", "Y": "nY"}

    np.random.seed(1)
    noise_dist["nX1"] = np.random.dirichlet(0.1 * np.ones(noise_states), size=1)[0].tolist()
    np.random.seed(2)
    noise_dist["nX2"] = np.random.dirichlet(0.1 * np.ones(noise_states), size=1)[0].tolist()
    np.random.seed(3)
    noise_dist["nW"] = np.random.dirichlet(0.1 * np.ones(noise_states), size=1)[0].tolist()
    np.random.seed(4)
    noise_dist["nY"] = np.random.dirichlet(0.1 * np.ones(noise_states), size=1)[0].tolist()


def set_ETT_NonId_graph(noise_states, latent_state):
    DAG_desc = "ETT_NonId"
    Complete_DAG_desc = "ETT_NonId"

    Complete_DAG = {}
    Complete_DAG["L"] = []
    Complete_DAG["M"] = []
    Complete_DAG["X1"] = ["M"]
    Complete_DAG["X2"] = ["L", "M", "X1"]
    Complete_DAG["W"] = ["X1", "X2"]
    Complete_DAG["Y"] = ["W", "L"]
    complete_labels = list(Complete_DAG.keys())

    Observed_DAG = {}
    Observed_DAG["X1"] = []
    Observed_DAG["X2"] = ["X1"]
    Observed_DAG["W"] = ["X1", "X2"]
    Observed_DAG["Y"] = ["W"]
    label_names = list(Observed_DAG.keys())

    interv_queries = [
        # {"obs": ["X2"], "interv": ["X1"], "expr": "P(X2|do(X1))"},
        # {"obs": ["Y"], "interv": ["W"], "expr": "P(Y|do(W))"},
        {"obs": ["W"], "interv": ["X1", "X2"], "expr": "P(W|do(X1,X2))"},
        {"obs": ["X2", "Y"], "interv": ["X1", "W"], "expr": "P(X2,Y)|do(X1,W)"},
        # {"obs": ["W", "Y"], "interv": ["X1", "X2"], "expr": "P(W,Y)|do(X1,X2)"}
    ]

    latent_conf = {"X1": ["M"], "X2": ["L", "M"], "W": [], "Y": ["L"]}
    confTochild = {"L": ["X2", "Y"], "M": ["X1", "X2"]}
    exogenous = {"X1": "nX1", "X2": "nX2", "W": "nW", "Y": "nY"}

    # counterfactual
    cf_intervene = {"X1": 0, "X2": 0}  # check if fractions do any better
    cf_observe = ["Y"]
    cf_evidence = {"X1p": 1, "X2p": 1}

    cflabel_names = ["X1", "X1p", "X2", "X2p", "W", "Y", "L", "M"]
    twin_map = {"X1p": "X1", "X1": "X1p", "X2p": "X2", "X2": "X2p"}

    Twin_Network = {}
    Twin_Network["X1"] = []
    Twin_Network["X2"] = []
    Twin_Network["L"] = []
    Twin_Network["M"] = []
    Twin_Network["X1p"] = ["M"]
    Twin_Network["X2p"] = ["X1p", "L", "M"]
    Twin_Network["W"] = ["X1", "X2"]
    Twin_Network["Y"] = ["W", "L"]
    cf_exogenous = {"X1p": "nX1", "X2p": "nX2", "W": "nW", "Y": "nY"}

    noise_params = {"nX1": (0.1, noise_states),
                    "nX2": (0.1, noise_states),
                    "nW": (0.1, noise_states),
                    "nY": (0.1, noise_states),
                    "L": (1, latent_state),
                    "M": (1, latent_state)}


    # mechanism training
    intervention_datavar = ["X1","W"]  # I cant concatenate different intvened variables distributions.

    train_mech_list=[
        {"mech": "X1", "parents": [], "intv": [], "compare": ["X1"]},
        {"mech": "X2", "parents": ["X1"], "intv": ["X1"], "compare": ["X1", "X2"]},
        {"mech": "W", "parents": ["X1", "X2"], "intv": [], "compare": ["X1", "X2", "W"]},   #it was "intv": ["X1"] but cant use that bcz there is no P(W|do(X1)) in P(V|do(X1,W))
        {"mech": "Y", "parents": ["W"], "intv": ["X1","W"], "compare": ["X1","X2", "W", "Y"]},  #!change intv w to x1 and w
    ]



    return DAG_desc, Complete_DAG_desc, Complete_DAG, complete_labels, Observed_DAG, label_names, interv_queries, latent_conf, \
           confTochild, exogenous, cf_intervene, cf_observe, cf_evidence, cflabel_names, twin_map, Twin_Network, cf_exogenous, noise_params, intervention_datavar, train_mech_list,


def set_minimal_graph(noise_states, latent_state):
    DAG_desc = "minimal_graph"

    Complete_DAG_desc = "minimal_graph"
    Complete_DAG = {}
    Complete_DAG["X"] = []
    Complete_DAG["Y"] = ["X"]
    complete_labels = list(Complete_DAG.keys())

    Observed_DAG = {}
    Observed_DAG["X"] = []
    Observed_DAG["Y"] = ["X"]
    label_names = list(Observed_DAG.keys())

    interv_queries = [
        {"obs": ["Y"], "interv": ["X"], "expr": "P(Y|do(X))"},
    ]

    latent_conf = {"X": [], "Y": []}
    confTochild = {}
    exogenous = {"X": "nX", "Y": "nY"}

    cf_intervene = {"X": 0}  # check if fractions do any better
    cf_observe = ["Y"]
    cf_evidence = {"Yp": 1}

    cflabel_names = ["X", "Xp", "Y", "Yp"]
    twin_map = {"Xp": "X", "X": "Xp", "Yp": "Y", "Y": "Yp"}

    Twin_Network = {}
    Twin_Network["X"] = []
    Twin_Network["Xp"] = []
    Twin_Network["Y"] = ["X"]
    Twin_Network["Yp"] = ["Xp"]
    cf_exogenous = {"Xp": "nX", "Y": "nY", "Yp": "nY"}

    noise_params = {"nX": (0.1, noise_states),
                    "nY": (0.1, noise_states), }

    return DAG_desc, Complete_DAG_desc, Complete_DAG, complete_labels, Observed_DAG, label_names, interv_queries, latent_conf, \
           confTochild, exogenous, cf_intervene, cf_observe, cf_evidence, cflabel_names, twin_map, Twin_Network, cf_exogenous, noise_params


def set_nonID_minimal_graph(noise_states, latent_state):
    DAG_desc = "nonID_minimal_graph"

    Complete_DAG_desc = "nonID_minimal_graph"
    Complete_DAG = {}
    Complete_DAG["L"] = []
    Complete_DAG["X"] = ["L"]
    Complete_DAG["Y"] = ["L", "X"]
    complete_labels = list(Complete_DAG.keys())

    Observed_DAG = {}
    Observed_DAG["X"] = []
    Observed_DAG["Y"] = ["X"]
    label_names = list(Observed_DAG.keys())

    interv_queries = [
        {"obs": ["Y"], "interv": ["X"], "expr": "P(Y|do(X))"},
    ]

    latent_conf = {"X": ["L"], "Y": ["L"]}
    confTochild = {"L": {"X", "Y"}}
    exogenous = {"X": "nX", "Y": "nY"}
    # exogenous = {}

    cf_intervene = {"X": 0}  # check if fractions do any better
    cf_observe = ["Y"]
    cf_evidence = {"Yp": 1}

    cflabel_names = ["L", "X", "Xp", "Y", "Yp"]
    twin_map = {"Xp": "X", "X": "Xp", "Yp": "Y", "Y": "Yp"}

    # twin network might be wrong!!!!
    Twin_Network = {}
    Twin_Network["L"] = []
    Twin_Network["X"] = []
    Twin_Network["Xp"] = ["L"]
    Twin_Network["Y"] = ["L", "X"]
    Twin_Network["Yp"] = ["L", "Xp"]
    cf_exogenous = {"Xp": "nX", "Y": "nY", "Yp": "nY"}
    # cf_exogenous = {}

    noise_params = {"nX": (0.1, noise_states),
                    "nY": (0.1, noise_states),
                    "L": (1, latent_state), }

    # noise_params = {"L": (1, latent_state)}

    # mechanism training
    intervention_datavar = ["X"]  # I cant concatenate different intvened variables distributions.
    train_mech_list=[
        {"mech": "X", "parents": [], "intv": [], "compare": ["X"]},
        {"mech": "Y", "parents": ["X"], "intv": ["X"], "compare": ["X", "Y"]}
    ]

    return DAG_desc, Complete_DAG_desc, Complete_DAG, complete_labels, Observed_DAG, label_names, interv_queries, latent_conf, \
           confTochild, exogenous, cf_intervene, cf_observe, cf_evidence, cflabel_names, twin_map, Twin_Network, cf_exogenous, noise_params, intervention_datavar, train_mech_list




def set_mnist_addition_graph_id(noise_states, latent_state):
    DAG_desc = "mnist_addition_graph"

    Complete_DAG_desc = "mnist_addition_graph"
    Complete_DAG = {}
    Complete_DAG["digit1"] = []
    Complete_DAG["digit2"] = ["digit1"]
    Complete_DAG["sign"] = ["digit1", "digit2"]
    Complete_DAG["result"] = ["digit1", "digit2", "sign"]
    complete_labels = list(Complete_DAG.keys())

    Observed_DAG = {}
    Observed_DAG["digit1"] = []
    Observed_DAG["digit2"] = ["digit1"]
    Observed_DAG["sign"] = ["digit1", "digit2"]
    Observed_DAG["result"] = ["digit1", "digit2", "sign"]
    label_names = list(Observed_DAG.keys())

    interv_queries = [
        {"obs": ["result"], "interv": ["digit1"], "expr": "P(result|do(digit1))"},
    ]

    latent_conf = {"digit1": [], "digit2": [], "sign":[], "result":[]}
    confTochild = {}
    exogenous = {"digit1": "ndigit1", "digit2": "ndigit2", "sign": "nsign", "result":"nresult" }

    cf_intervene = {"digit1":0, "sign":1}  # check if fractions do any better
    cf_observe = []
    cf_evidence = {}

    cflabel_names = []
    twin_map = {}

    # twin network might be wrong!!!!
    Twin_Network = {}
    cf_exogenous = {}

    noise_params = {"ndigit1": (0.1, noise_states),
                    "ndigit2": (0.1, noise_states),
                    "nsign": (0.1, noise_states),
                    "nresult": (0.1, noise_states)
                    }


    # mechanism training
    intervention_datavar = []  # I cant concatenate different intvened variables distributions.
    train_mech_list=[
        {"mech": "digit1", "parents": [], "intv": [], "compare": ["digit1"]},
        {"mech": "digit2", "parents": ["digit1"], "intv": [], "compare": ["digit1", "digit2"]},
        {"mech": "sign", "parents": ["digit1","digit2"], "intv": [], "compare": ["digit1", "digit2", "sign"]},
        {"mech": "result", "parents": ["digit1", "digit2", "sign"], "intv": [], "compare": ["digit1", "digit2", "sign", "result"]}
    ]

    return DAG_desc, Complete_DAG_desc, Complete_DAG, complete_labels, Observed_DAG, label_names, interv_queries, latent_conf, \
           confTochild, exogenous, cf_intervene, cf_observe, cf_evidence, cflabel_names, twin_map, Twin_Network, cf_exogenous, noise_params, intervention_datavar, train_mech_list





def set_mnist_addition_graph(noise_states, latent_state):
    DAG_desc = "mnist_addition_graph"

    Complete_DAG_desc = "mnist_addition_graph"
    Complete_DAG = {}
    Complete_DAG["Uthick"] = []
    Complete_DAG["Ucolor"] = []
    Complete_DAG["digit1"] = ["Uthick"]
    Complete_DAG["digit2"] = ["Uthick", "Ucolor", "digit1"]
    Complete_DAG["sign"] = ["digit1", "digit2"]
    # Complete_DAG["result"] = ["Ucolor", "digit1", "digit2", "sign"]
    Complete_DAG["result"] = ["digit2"]
    complete_labels = list(Complete_DAG.keys())

    Observed_DAG = {}
    Observed_DAG["digit1"] = []
    Observed_DAG["digit2"] = ["digit1"]
    Observed_DAG["sign"] = ["digit1", "digit2"]
    Observed_DAG["result"] = ["digit2"]
    label_names = list(Observed_DAG.keys())

    interv_queries = [
        {"obs": ["result"], "interv": ["digit1"], "expr": "P(result|do(digit1))"},
    ]

    latent_conf = {"digit1": ["Uthick"], "digit2": ["Uthick","Ucolor"], "sign":[], "result":[]}
    confTochild = {"Uthick": ["digit1", "digit2"], "Ucolor":["digit2"]}
    exogenous = {"digit1": "ndigit1", "digit2": "ndigit2", "sign": "nsign", "result":"nresult" }

    cf_intervene = {"digit1":0, "sign":1}  # check if fractions do any better
    cf_observe = []
    cf_evidence = {}

    cflabel_names = []
    twin_map = {}

    # twin network might be wrong!!!!
    Twin_Network = {}
    cf_exogenous = {}

    noise_params = {"ndigit1": (0.1, noise_states),
                    "ndigit2": (0.1, noise_states),
                    "nsign": (0.1, noise_states),
                    "nresult": (0.1, noise_states),
                    "Uthick": (1, latent_state),
                    "Ucolor": (1, latent_state)
                    }


    # mechanism training
    intervention_datavar = []  # I cant concatenate different intvened variables distributions.
    train_mech_list=[
        {"mech": "digit1", "parents": [], "intv": [], "compare": ["digit1"]},
        {"mech": "digit2", "parents": ["digit1"], "intv": [], "compare": ["digit1", "digit2"]},
        {"mech": "sign", "parents": ["digit1","digit2"], "intv": [], "compare": ["digit1", "digit2", "sign"]},
        {"mech": "result", "parents": ["digit2"], "intv": [], "compare": ["digit2"]}
    ]

    return DAG_desc, Complete_DAG_desc, Complete_DAG, complete_labels, Observed_DAG, label_names, interv_queries, latent_conf, \
           confTochild, exogenous, cf_intervene, cf_observe, cf_evidence, cflabel_names, twin_map, Twin_Network, cf_exogenous, noise_params, intervention_datavar, train_mech_list



def set_nonID_frontdoor_graph(noise_states, latent_state):
    DAG_desc = "nonID_frontdoor_graph"

    Complete_DAG_desc = "nonID_frontdoor_graph"
    Complete_DAG = {}
    Complete_DAG["L1"] = []
    Complete_DAG["L2"] = []
    Complete_DAG["X"] = ["L1"]
    Complete_DAG["Z"] = ["L2", "X"]
    Complete_DAG["Y"] = ["L1","L2", "Z"]
    complete_labels = list(Complete_DAG.keys())

    Observed_DAG = {}
    Observed_DAG["X"] = []
    Observed_DAG["Z"] = ["X"]
    Observed_DAG["Y"] = ["Z"]
    label_names = list(Observed_DAG.keys())

    interv_queries = [
        {"obs": ["Z"], "interv": ["X"], "expr": "P(Z|do(X))"},
        {"obs": ["Y"], "interv": ["X"], "expr": "P(Y|do(X))"},
        {"obs": ["Y"], "interv": ["Z"], "expr": "P(Y|do(Z))"},
    ]

    latent_conf = {"X": ["L1"], "Z": ["L2"], "Y": ["L1, L2"]}
    confTochild = {"L1": ["X", "Y"], "L2":["Z","Y"]}
    exogenous = {"X": "nX", "Z":"nZ", "Y": "nY"}

    cf_intervene = {"X": 0}  # check if fractions do any better
    cf_observe = []
    cf_evidence = {}

    cflabel_names = []
    twin_map = {}

    # twin network might be wrong!!!!
    Twin_Network = {}
    cf_exogenous = {}

    noise_params = {"nX": (0.1, noise_states),
                    "nZ": (0.1, noise_states),
                    "nY": (0.1, noise_states),
                    "L1": (1, latent_state),
                    "L2": (1, latent_state), }

    # mechanism training
    intervention_datavar = ["Z"],  # I cant concatenate different intvened variables distributions.
    train_mech_list=[
    {"mech": "X", "parents": [], "intv": [], "compare": ["X"]},
    {"mech": "Z", "parents": ["X"], "intv": [], "compare": ["X", "Z"]},
    {"mech": "Y", "parents": ["Z"], "intv": ["Z"], "compare": ["X", "Z", "Y"]}
    ]

    # G_hid_dims = [16, 24, 20, 16, 8],  # 5 layers
    # D_hid_dims = [12, 20, 15, 5],  # 5 layers

    return DAG_desc, Complete_DAG_desc, Complete_DAG, complete_labels, Observed_DAG, label_names, interv_queries, latent_conf, \
           confTochild, exogenous, cf_intervene, cf_observe, cf_evidence, cflabel_names, twin_map, Twin_Network, cf_exogenous, noise_params, intervention_datavar, train_mech_list



def set_single_variable_graph(noise_states, latent_state):
    DAG_desc = "single_variable_graph"

    Complete_DAG_desc = "single_variable_graph"
    Complete_DAG = {}
    Complete_DAG["X1"] = []
    Complete_DAG["X2"] = ["X1"]
    Complete_DAG["X3"] = ["X2"]
    Complete_DAG["X4"] = ["X3"]
    Complete_DAG["X5"] = ["X4"]
    complete_labels = list(Complete_DAG.keys())

    Observed_DAG = {}
    Observed_DAG["X1"] = []
    Observed_DAG["X2"] = ["X1"]
    Observed_DAG["X3"] = ["X2"]
    Observed_DAG["X4"] = ["X3"]
    Observed_DAG["X5"] = ["X4"]
    label_names = list(Observed_DAG.keys())
    num_labels = len(label_names)

    interv_queries = [

    ]

    latent_conf = {"X1": [], "X2": [], "X3": [], "X4": [], "X5": []}
    confTochild = {}
    exogenous = {"X1": "nX1", "X2": "nX2", "X3": "nX3", "X4": "nX4", "X5": "nX5"}

    cf_intervene = {"X": 0}  # check if fractions do any better
    cf_observe = []
    cf_evidence = {}

    cflabel_names = ["X", "Xp"]
    twin_map = {"Xp": "X", "X": "Xp"}

    Twin_Network = {}
    Twin_Network["X"] = []
    Twin_Network["Xp"] = []
    cf_exogenous = {"Xp": "nX"}

    noise_dist = {}
    np.random.seed(1)
    noise_dist["nX1"] = np.random.dirichlet(0.1 * np.ones(noise_states), size=1)[0].tolist()
    np.random.seed(2)
    noise_dist["nX2"] = np.random.dirichlet(0.1 * np.ones(noise_states), size=1)[0].tolist()
    np.random.seed(3)
    noise_dist["nX3"] = np.random.dirichlet(0.1 * np.ones(noise_states), size=1)[0].tolist()
    np.random.seed(4)
    noise_dist["nX4"] = np.random.dirichlet(0.1 * np.ones(noise_states), size=1)[0].tolist()
    np.random.seed(5)
    noise_dist["nX5"] = np.random.dirichlet(0.1 * np.ones(noise_states), size=1)[0].tolist()

    return DAG_desc, Complete_DAG_desc, Complete_DAG, complete_labels, Observed_DAG, label_names, interv_queries, latent_conf, \
           confTochild, exogenous, cf_intervene, cf_observe, cf_evidence, cflabel_names, twin_map, Twin_Network, cf_exogenous, noise_dist

# use networkX library?
# set_nonID_graph()
# set_frontdoor_with_latents()
# set_frontdoor_wo_latents()
# set_backdoor_wi_latents()
# set_nonID_H_graph()

# set_ETT_Id_graph()
# set_ETT_Id_graph_nolatents()
# set_ETT_NonId_graph()
# set_minimal_graph()
