import copy
import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm

from modularTraining.constantFunctions import get_dataloaders, Parameters, get_prediction
from utils.constant import ATTRIBUTES
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datamodules.celebadatamodule import CelebADataModule
from lightningmodules.classification import Classification
import numpy as np

import torch
from pytorch_lightning import Trainer

################################ Classification
from dataclasses import dataclass
import os, os.path as osp
from typing import Any, ClassVar, Dict, List, Optional


from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import modularTraining.constant_paths as const






def get_label_distribution(data_loader):
    found = {}
    for lb in attributes:
        found[lb] = 0

    for i, (img, lbl) in tqdm(enumerate(data_loader)):
        predictions = get_prediction(classifier, trainer, img)
        for iter,pred in enumerate(predictions):

            for lb in pred:
                found[lb] += 1

            if 'Young' in pred:
                print('Found Young')
                plt.imshow(img[iter].permute(1,2,0))
                plt.show()
                print('Shown Young')


        # break

    print('Attribute found')
    # print(found)
    # for lb in found:
    #     found[lb] = found[lb] / (len(data_loader) * 64) * 100
    found = dict(sorted(found.items(), key=lambda item: item[1]))
    print(found)

    return found


def get_idata(name):
    save_folder = f'/{const.project_root}/modularCelebA/images256/{name}'

    file = f'{save_folder}/images_10k_I_doMale.pkl'
    with open(file, 'rb') as f:
        intv_images = pickle.load(f)

    file = f'{save_folder}/labels_10k_I_doMale.pkl'
    with open(file, 'rb') as f:
        intv_labels = pickle.load(f)

    return intv_labels, intv_images


def get_train_test_loaders(dom_name,split_t=0.95, split_v=0.9):
    folder = f'/{const.project_root}/modularCelebA/images256'
    file = f'{folder}/8_attribute_10k_celeba_{dom_name}.pkl'
    with open(file, 'rb') as f:
        domain_dataset = pickle.load(f)

    folder = f'/{const.project_root}/modularCelebA/images256'
    file = f'{folder}/images_10k_celeba_{dom_name}.pkl'
    with open(file, 'rb') as f:
        images = pickle.load(f)

    domain_dataset['I'] = images['I']
    dom_age = domain_dataset['Young'].reshape(-1, 1).type(torch.FloatTensor)
    dom_images = domain_dataset['I']

    trainloader, validloader, testloader = get_dataloaders(dom_age, dom_images,split_t=0.95, split_v=0.9)

    return domain_dataset, trainloader, validloader, testloader




if __name__ == '__main__':
    # loading the classifier
    attributes = list(ATTRIBUTES.values())
    config = Parameters()
    checkpoint = torch.load(config.inference_param.ckpt_path)
    classifier = Classification(config.inference_param)
    classifier.load_state_dict(checkpoint["state_dict"])
    print('Classifier loaded')
    trainer = Trainer(devices=config.hparams.gpu, limit_train_batches=0, limit_val_batches=0)
    print('loaded')

    # loading interventional data
    l1, i1 = get_idata('fake5000')  # getting  1st 5000k data samples
    l2, i2 = get_idata('fake10000')  # getting 2nd 5000k data samples
    intv_labels = {}
    intv_labels['Male'] = np.concatenate([l1['Male'], l2['Male']])
    intv_labels['Young'] = np.concatenate([l1['Young'], l2['Young']])
    intv_images = np.concatenate([i1['I'], i2['I']])
    intv_age = np.array(intv_labels['Young'], dtype='float32').reshape(-1, 1)
    # trainloaderI, _, _ = get_dataloaders(intv_age, intv_images, split_t=0.95, split_v=0.99)


    intv_dataset = copy.deepcopy(intv_labels)
    for key in intv_dataset:
        intv_dataset[key] = torch.tensor(intv_dataset[key])
    intv_dataset['I'] = intv_images
    male = 0
    young = 0
    retfo = (intv_dataset['Male'] == male) & (intv_dataset['Young'] == young)
    female_old = intv_dataset['Young'][retfo].reshape(-1, 1).type(torch.FloatTensor)
    female_image = intv_dataset['I'][retfo]
    m0y0_trainloaderI, _, _ = get_dataloaders(female_old, female_image,split_t=0.95, split_v=0.9)
    #

    #
    # print('label distribution in female old interventional data')
    # distI= get_label_distribution(m0y0_trainloaderI)

    exists =  {}
    for lb in attributes:
        exists[lb] = 0


    pred1 = get_prediction(classifier, trainer, female_image)

    # data_list = []
    # for img in female_image:
    #     lbl = torch.zeros(40, 1)
    #     data_list.append([img, lbl])
    #
    # predict_loader = DataLoader(dataset=data_list, batch_size=1, shuffle=False)
    # prediction = trainer.predict(classifier, predict_loader)  # without fine-tuning
    # pred1 = []
    # for idx, data_input in enumerate(prediction):
    #     pred = data_input[2][0]
    #     pred1.append(pred)



    for att in pred1:
        for lb in att:
            exists[lb] += 1





    print('Attribute increased')
    print(exists)
    for lb in exists:
        exists[lb] = exists[lb] / (female_image.shape[0]) * 100
    increased = dict(sorted(exists.items(), key=lambda item: item[1]))
    print(increased)



