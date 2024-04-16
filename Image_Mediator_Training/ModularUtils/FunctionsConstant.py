import copy
import os

import numpy as np
import torch
import torchvision.transforms as transforms
import pickle

from matplotlib import pyplot as plt

from ModularUtils.ControllerConstants import get_multiple_labels_fill
from ModularUtils.DigitImageGeneration.morphomnist import io
from ModularUtils.FunctionsTraining import get_training_variables
from matplotlib.cm import get_cmap
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


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





# load_checkpoint(Exp.SAVED_PATH, label_generators, [G_optimizers], discriminators, D_optimizers, Exp.learning_rate)


# compare two nn model


def initialize_results(Exp, cur_hnodes):
    tvd_diff = {}
    kl_diff = {}
    # for hn, cur_mechs in cur_hnodes.items():
    #
    #     tvd_diff = {}
    #     kl_diff = {}
    #
    #     cur_mechs= [lb for lb in cur_mechs if lb not in Exp.image_labels]
    #     if len(cur_mechs)==0:
    #         continue
    #
    #     query= getdoKey(cur_mechs, {})
    #     tvd_diff[query] = []
    #     kl_diff[query] = []

        # all_var= copy.deepcopy(cur_mechs)
        # if "X1" in all_var:
        #     all_var.remove("X1")
        # query = getdoKey(all_var, {"X1":0})
        # tvd_diff[hn][query] = []
        # kl_diff[hn][query] = []
        #
        # query = getdoKey(all_var, {"X1": 1})
        # tvd_diff[hn][query] = []
        # kl_diff[hn][query] = []

    # compare_Var = [lb for lb in Exp.label_names if lb not in Exp.image_labels+Exp.rep_labels]
    # obs_query= getdoKey(compare_Var, {})
    # tvd_diff[obs_query] = []
    # kl_diff[obs_query] = []
        #
    for query_list in Exp.interv_queries:

        for intv in query_list["intervs"]:
            query = getdoKey(query_list["obs"], intv)
            tvd_diff[query] = []
            kl_diff[query] = []

        # tvd_diff[query_list["expr"]] = []
        # kl_diff[query_list["expr"]] = []

        #
    for query in Exp.cf_queries:
        tvd_diff[query["expr"]] = []
        kl_diff[query["expr"]] = []

        # KeyError: 'P(X1X2WYdigit1Ydigit2YcolorYthick|do_[])'

    # if True in Exp.load_which_models.values() or train_no>0 :
    if True in Exp.load_which_models.values() :
        print("loading previous tvd diffs")
        for dist in tvd_diff:
            if os.path.exists(Exp.LOAD_MODEL_PATH + "/tvd/" + dist):
                tvd_diff[dist] = torch.load(Exp.LOAD_MODEL_PATH + "/tvd/" + dist).tolist()
                kl_diff[dist] = torch.load(Exp.LOAD_MODEL_PATH + "/kl/" + dist).tolist()


    return tvd_diff, kl_diff



############ Dataset functions ############

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

# '/local/scratch/a/rahman89/PycharmProjects/conditional-DCGAN/SAVED_EXPERIMENTS/imageMediator/preprocessed_dataset/intv0D.pkl'

def get_Imagedataset(Exp, intv_no, digit):
    # filext = "Ydigit1images.gz" if digit == "ImgYdigit1" else "Ydigit2images.gz"
    # labels_data = io.load_idx(f"{Exp.file_roots[0]}Ydigit2labels.gz")
    # loaded_images = io.load_idx(f"{Exp.file_roots[0]}{digit}.gz")
    # labels_data = io.load_idx(f"{Exp.file_roots[0]}{digit}labels.gz")

    if digit=="ImgYdigit1":
        # loaded_images = io.load_idx(f"{Exp.file_roots[0]}Ydigit1images.gz")
        # labels_data = io.load_idx(f"{Exp.file_roots[0]}Ydigit1labels.gz") + "Ydigit1images.gz"
        loaded_images = io.load_idx(f"{Exp.file_roots+ str(intv_no)}Ydigit1images.gz")
        labels_data = io.load_idx(f"{Exp.file_roots+ str(intv_no)}Ydigit1labels.gz")

    elif digit=="ImgYdigit2":
        loaded_images = io.load_idx(f"{Exp.file_roots[0]}Ydigit2images.gz")
        labels_data = io.load_idx(f"{Exp.file_roots[0]}Ydigit2labels.gz")


    transform = transforms.Compose([transforms.ToPILImage(),
                                    # transforms.Scale(Exp.IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(
                                    # 0.5, 0.5, 0.5))

                                    ])  # Try with normalizing!  need to normalize too since I am using tanh?



    digit_images = [torch.unsqueeze(transform(img), dim=0).to(Exp.DEVICE) for img in loaded_images]
    digit_images = torch.cat(digit_images, 0)


    # a new dataset structure for images with only observations and images.

    # for id, img in enumerate(digit_images):
    #     if id == 50:
    #         break
    #     if id<50:
    #         continue
    #
    #     # if labels_data[id][2]==1:
    #     #     continue
    #     print(labels_data[id])
    #     imggg1 = img.permute(1, 2, 0).detach().cpu().numpy()
    #     fig, ax = plt.subplots()
    #     plt.imshow(imggg1)
    #     plt.show()



    return digit_images


def load_image_dataset(Exp, cur_hnodes):
    image_data_dict = {}
    for hnode, cur_mechs in cur_hnodes.items():
        # image_data_dict={}
        # for dno in range(Exp.num_datasets):
        for dno, intv in enumerate(Exp.Data_intervs):
            all_compare_Var, compare_Var, intervened_Var, real_labels_vars = get_training_variables(Exp, cur_mechs, dno, {})

            # ---------load image dataset  intv0X0

            image_dataset = []
            # for dno in range(Exp.num_datasets):  # number of interventional datasets including observations for this specific hnode
            for mech in all_compare_Var:
                # if set(mech) & set(Exp.image_labels) != set():
                if mech in Exp.image_labels:
                    digit_images = get_Imagedataset(Exp, 0, "ImgYdigit1")
                    image_dataset.append(digit_images)
            if len(image_dataset):
                image_data_dict[asKey(Exp.Data_intervs[dno])] = torch.cat(image_dataset, 1).to(Exp.DEVICE)

    return image_data_dict

def load_label_dataset(Exp, image_data_dict, label_generators, cur_hnodes, bayes_graph=None):  #get all datasets despite any hnodes

    dataset_dict = {}

    for dno in range(Exp.num_datasets):
    # for dno, intv in enumerate(Exp.Data_intervs):
        intv= Exp.Data_intervs[dno]
        all_compare_Var, compare_Var, intervened_Var, real_labels_vars = get_training_variables(Exp, Exp.label_names, dno, intv)

        # load datasets without images

        dataset_dict[asKey(Exp.Data_intervs[dno])]={}
        repdata_dict ={}
        # need change here too.
        each_dataset = []
        rep_dataset = []
        for label in real_labels_vars:
            # if label not in compare_Var:
            if label not in Exp.rep_labels:
                each_dataset.append(get_dataset(Exp, label, dno))
            # else:
        # for hnode, cur_mechs in cur_hnodes.items():
        #     all_compare_Var, compare_Var, intervened_Var, real_labels_vars = get_training_variables(Exp, cur_mechs, dno, {})

        # ---- Load latent representaiton ----#
        for rep in Exp.rep_labels:

                parent= Exp.Observed_DAG[rep][1]   #Taking only digit as parent here.
                y_discrete=get_dataset(Exp, parent, dno)
                dim_list = [Exp.label_dim[parent]]
                label_onehots = get_multiple_labels_fill(Exp, y_discrete.view(-1, 1), dim_list, isImage_labels=False,)
                image_values= image_data_dict[asKey({})]
                # image_latents = label_generators[rep](Exp, image_values, label_onehots , dim_list, isOnehot=False, isLatent=True)
                image_latents = label_generators[rep](Exp, image_values, label_onehots , dim_list, isLatent=True)

                # out_image = label_generators[rep](Exp, image_values[0:1], label_onehots[0:1], dim_list, isOnehot=False, isLatent=False)  #printing one image
                # img = out_image[0].permute(1, 2, 0).detach().cpu().numpy()
                # plot_trained_digits(1, 1, [img], f'Real')

                rep_dataset.append(image_latents)
                dataset_dict[asKey(Exp.Data_intervs[dno])]["rep"] = torch.cat(rep_dataset, 1).to(Exp.DEVICE)
        # ----  x -------

        dataset_dict[asKey(Exp.Data_intervs[dno])]["obs"] = torch.cat(each_dataset, 1).to(Exp.DEVICE)

        # if dno==0 and bayes_graph!=None:
        #     Exp.bayesNet = prepare_bn(Exp, bayes_graph, dataset_dict[asKey(Exp.Data_intervs[dno])]["obs"], load_scm=1)
        #     print(Exp.bayesNet.cpt('Y'))

        # break  #for only observational data

    return dataset_dict


def save_datasets(SAVE_DATASET, label_save_dir, feature, true_data):
    if SAVE_DATASET == False:
        return

    for label in true_data:
        file_name = label_save_dir + label  + ".pkl"
        with open(file_name, 'wb') as fp:
            pickle.dump(np.array(true_data[label]), fp)
        print(file_name, " saved")


################ Plotting ###########
def plot_lines(title, ylabel, diff_list , xaxis, labels, dashed, dashdot, tvd_error=None, save_plot=None, path=None):


    # plt.rc('font', size=12)
    # SMALL_SIZE = 8

    plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
    plt.rc('legend', fontsize=10.5)  # legend fontsize

    # name = "Dark2"
    name = "tab10"
    # name = "tab20"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    base = list(cmap.colors)  # type: list
    colors =[base[2], base[-1], base[3]]

    linestyle= ['solid', 'dashed', 'dotted']
    for i in range (len(diff_list)):
        if labels[i] in dashed:  # algorithm result: P(V), P(Y|do(X)) Benchmark ncm_P(V), ncm_P(Y|do(X))
            col=colors[2]
        elif labels[i] in dashdot:
            col=colors[1]
        else:
            col=colors[0]

        cid=i
        if len(dashed)!=0 and len(dashdot)!=0:
            cid=int(i/3)
        elif len(dashed)!=0:
            cid = int(i / 2)

        plt.plot(xaxis, diff_list[i], label=labels[i], linestyle=linestyle[cid], color= col)  #'solid', 'dashed', 'dashdot', 'dotted'
        y, e= np.array(diff_list[i]), np.array(tvd_error[i])
        plt.fill_between(xaxis, y-e, y+e, color= col, alpha=0.2)  #'solid', 'dashed', 'dashdot', 'dotted'


    # for i in range (len(diff_list)):
    #     if labels[i] in dashed:  # algorithm result: P(V), P(Y|do(X)) Benchmark ncm_P(V), ncm_P(Y|do(X))
    #         linestyle="dashed"
    #     elif labels[i] in dashdot:
    #         linestyle="dotted"
    #     else:
    #         linestyle="solid"
    #
    #     cid=i
    #     if len(dashed)!=0 and len(dashdot)!=0:
    #         cid=int(i/3)
    #     elif len(dashed)!=0:
    #         cid = int(i / 2)
    #
    #     plt.plot(xaxis, diff_list[i], label=labels[i], linestyle=linestyle, color= colors[cid])  #'solid', 'dashed', 'dashdot', 'dotted'
    #     y, e= np.array(diff_list[i]), np.array(tvd_error[i])
    #     plt.fill_between(xaxis, y-e, y+e, color= colors[cid], alpha=0.2)  #'solid', 'dashed', 'dashdot', 'dotted'

    # plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend()

    # matplotlib.rcParams.update({'font.size': 12})

    if save_plot==True:
        plt.savefig(path+"/"+title+'.png', bbox_inches='tight')

    ax = plt.subplot(111)
    ncol=3
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=ncol, fancybox=True, shadow=False)


    plt.ylim([0, 0.8])
    plt.grid(True)
    plt.show()

