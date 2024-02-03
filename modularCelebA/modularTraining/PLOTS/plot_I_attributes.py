import os


def plot_image_ara(img_ara, folder=None, title=None):
    rows = img_ara.shape[0]
    cols = img_ara.shape[1]

    print(rows, cols)

    f, axarr = plt.subplots(rows, cols, figsize=(cols, rows), squeeze=False)
    for c in range(cols):

        if c == 0:
            # axarr[r, c].get_yaxis().set_ticks([ '','x', ''])
            # axarr[r, c].get_yaxis().set_ticks([x for x in range(-10,11)])
            axarr[0, c].set_yticklabels(['', '', 'Sex=0 \nEye=1'], rotation=0, fontsize=12)
            axarr[1, c].set_yticklabels(['', '', 'Sex=1 \n Eye=1'], rotation=0, fontsize=12)

        for r in range(rows):
            if c > 0:
                axarr[r, c].get_yaxis().set_ticks([])

            axarr[r, c].get_xaxis().set_ticks([])
            img = img_ara[r][c].cpu().detach().numpy()
            # img= np.transpose(img, (1,2,0))
            axarr[r, c].imshow(img)

        f.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    if folder == None:

        plt.show()
    else:
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f'{folder}/{title}.png', bbox_inches='tight')

    plt.close()


import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import copy
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from modularTraining.constantFunctions import get_train_test_loaders, get_idata, denorm
from modularTraining.constantFunctions import get_sexWithAge

from Classifiers.mobileNet.model import MobileNet
from torch import nn
from Classifiers.mobileNet.train_valid_test import test

from modularTraining.constantFunctions import get_dataloaders
from Classifiers.mobileNet.train_valid_test import train, validation

exp_name = "sex_eyeglass"
targetAtt = 'Eyeglasses'

intv_labels = {}
folder = f'/{const.project_root}/modularCelebA/{exp_name}'
male_list = []
young_list = []
img_list = []

plt_img = []

st = 160
en = st + 16
for m, y in zip([0, 0, 1, 1], [0, 1, 0, 1]):
    # ll, ii = get_idata(f'fakeIdom{m}y{y}/images_Idom{m}y{y}.pkl', f'fakeIdom{m}y{y}/labels_Idom{m}y{y}.pkl')  #getting  m0y0 data samples
    ll, ii = get_idata(folder, f'fakeIdom{m}e{y}/images_Idom{m}e{y}.pkl',f'fakeIdom{m}e{y}/labels_Idom{m}e{y}.pkl')  # getting  m0e0 data samples

    plt_img.append(denorm(ii['I'][st:en]).permute(0, 2, 3, 1).unsqueeze(0))

male_list.append(ll['Male'])
young_list.append(ll['Young'])
img_list.append(ii['I'])

#


img_ara = torch.cat(plt_img)
plot_image_ara(img_ara)
plot_image_ara(img_ara, f'/{const.project_root}/modularCelebA/modularTraining/PLOTS', f"joint_sample{st}_{en}")
