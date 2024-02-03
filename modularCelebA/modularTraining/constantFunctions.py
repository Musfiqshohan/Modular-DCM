import copy

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import copy

from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

import copy
import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm

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
import modularTraining.constant_paths as const


def get_transform(image_size, isTensor=False):
    if isTensor == False:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(image_size),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    else:
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        return transform

    return transform


@dataclass
class Hparams:
    """Hyperparameters of for the run"""

    # wandb parameters
    wandb_project: str = "classif_celeba"
    wandb_entity: str = "rahman89"  # name of the project
    save_dir: str = osp.join(os.getcwd())  # directory to save wandb outputs
    weights_path: str = osp.join(os.getcwd(), "weights")

    # train or predict
    train: bool = True
    predict: bool = False

    gpu: int = 1
    fast_dev_run: bool = False
    limit_train_batches: float = 1.0
    val_check_interval: float = 0.5


@dataclass
class TrainParams:
    """Parameters to use for the model"""
    model_name: str = "vit_small_patch16_224"
    pretrained: bool = True
    n_classes: int = 40
    lr: int = 0.00001


@dataclass
class DatasetParams:
    """Parameters to use for the model"""
    # datamodule
    num_workers: int = 2  # number of workers for dataloadersint
    # root_dataset      : Optional[str] = osp.join(os.getcwd(), "assets")   # '/kaggle/working'
    # root_dataset      : Optional[str] = osp.join(os.getcwd(), "assets", "inputs")   # '/kaggle/working'
    root_dataset: Optional[str] = f"{const.dataset_root}/celeba/"
    # root_dataset      : Optional[str] = "/local/scratch/a/rahman89/CelebAMask-HQ"
    batch_size: int = 1  # batch_size
    input_size: tuple = (224, 224)  # image_size


@dataclass
class InferenceParams:
    """Parameters to use for the inference"""
    model_name: str = "vit_small_patch16_224"
    pretrained: bool = True
    n_classes: int = 40
    # ckpt_path: Optional[str] = osp.join(os.getcwd(), "weights", "ViTsmall.ckpt")
    ckpt_path: Optional[str] = osp.join(
        f"/{const.project_root}/interfacegan/AttributeClassification", "weights", "ViTsmall.ckpt")
    output_root: str = osp.join(os.getcwd(), "output")
    lr: int = 0.00001


@dataclass
class SVMParams:
    """Parameters to edit for SVM training"""
    json_file: str = "outputs_stylegan/stylegan3/scores_stylegan3.json"
    np_file: str = "outputs_stylegan/stylegan3/z.npy"
    output_dir: str = "trained_boundaries_z_sg3"
    latent_space_dim: int = 512
    equilibrate: bool = False


@dataclass
class Parameters:
    """base options."""

    hparams: Hparams = Hparams()
    data_param: DatasetParams = DatasetParams()
    train_param: TrainParams = TrainParams()
    inference_param: InferenceParams = InferenceParams()
    svm_params: SVMParams = SVMParams()

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance


class celeba(
    Dataset):  # image and labels from image paths; not doing any transformation bcz input images are already transformed
    def __init__(self, labels, images, image_size=224, doNormalize=False):
        self.labels = labels
        self.images = images

        # if doNormalize==False:
        # 	self.transform = transforms.Compose([transforms.Resize((image_size, image_size))])
        # else:
        # 	self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),
        # 										 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # 										 ])

        print('Trainsforming to image size ', image_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # image_tensor = self.transform(self.images[idx])
        image_tensor = self.images[idx]
        image_label = torch.Tensor(self.labels[idx])

        return image_tensor, image_label


# split data into train, valid, test set 7:2:1
# split_t=0.7
# split_v=0.9
def get_dataloaders(labels, images, split_t=0.7, split_v=0.9, img_size=224,
                    doNormalize=False):  # normalizing in (-1,1) and resizing images

    indices = list(range(len(labels)))
    split_train = int(len(indices) * split_t)
    split_valid = int(len(indices) * split_v)
    print(split_train, split_valid, len(indices))

    dataset = celeba(labels, images, image_size=img_size, doNormalize=doNormalize)

    train_idx, valid_idx, test_idx = indices[:split_train], indices[split_train:split_valid], indices[split_valid:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # print(len(train_idx),len(valid_idx), len(test_idx) )

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, sampler=train_sampler)
    validloader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler)
    testloader = torch.utils.data.DataLoader(dataset, sampler=test_sampler)

    return trainloader, validloader, testloader


def get_train_test_loaders(targetAtt, lbfile, imgfile):
    # folder= f'/{const.project_root}/modularCelebA/images256'
    # file =f'{folder}/8_attribute_10k_celeba_{dom_name}.pkl'
    # folder= f'/{const.project_root}/modularCelebA/images256'
    # file =f'{folder}/images_10k_celeba_{dom_name}.pkl'

    with open(lbfile, 'rb') as f:
        domain_dataset = pickle.load(f)

    with open(imgfile, 'rb') as f:
        images = pickle.load(f)

    domain_dataset['I'] = images['I']
    dom_age = domain_dataset[targetAtt].reshape(-1, 1).type(torch.FloatTensor)
    dom_images = domain_dataset['I']

    trainloader, validloader, testloader = get_dataloaders(dom_age, dom_images)

    return domain_dataset, trainloader, validloader, testloader


# Doing for domain 1

def get_sex_loaders(domain_dataset, needSameSize=False):
    retm = (domain_dataset['Male'] == 1)
    male_age = domain_dataset['Young'][retm].reshape(-1, 1)
    male_image = domain_dataset['I'][retm]

    retf = (domain_dataset['Male'] == 0)
    female_age = domain_dataset['Young'][retf].reshape(-1, 1)
    female_image = domain_dataset['I'][retf]

    if needSameSize == True:
        msz = min(male_age.shape[0], female_age.shape[0])
        male_age = male_age[0:msz]
        female_age = female_age[0:msz]

        male_image = male_image[0:msz]
        female_image = female_image[0:msz]

    male_trainloader, male_validloader, male_testloader = get_dataloaders(male_age, male_image)
    female_trainloader, female_validloader, female_testloader = get_dataloaders(female_age, female_image)

    return male_testloader, female_testloader


def get_sexWithAge(domain_dataset, att1, val1, att2, val2, doNormalize=False):
    retfo = (domain_dataset[att1] == val1) & (domain_dataset[att2] == val2)
    att1_att2_lb = domain_dataset[att2][retfo].reshape(-1, 1).type(torch.FloatTensor)
    att1_att2_image = domain_dataset['I'][retfo]
    oldfemale_trainloader, oldfemale_validloader, rare_testloader = get_dataloaders(att1_att2_lb, att1_att2_image,
                                                                                    split_t=0., split_v=0.,
                                                                                    doNormalize=doNormalize)
    return rare_testloader



def get_one_attribute_loader(domain_dataset, att1, val1, att2, doNormalize=False):
    retfo = (domain_dataset[att1] == val1)
    att1_lb = domain_dataset[att2][retfo].reshape(-1, 1).type(torch.FloatTensor)
    att1_image = domain_dataset['I'][retfo]
    oldfemale_trainloader, oldfemale_validloader, rare_testloader = get_dataloaders(att1_lb, att1_image,
                                                                                    split_t=0., split_v=0.,
                                                                                    doNormalize=doNormalize)
    return rare_testloader



def get_idata(folder, imgfile, lbfile):
    save_folder = folder

    file = f'{save_folder}/{imgfile}'
    with open(file, 'rb') as f:
        intv_images = pickle.load(f)

    file = f'{save_folder}/{lbfile}'
    with open(file, 'rb') as f:
        intv_labels = pickle.load(f)

    return intv_labels, intv_images


def get_prediction(classifier, trainer, images):
    transform = transforms.Compose(
        [
            # transforms.ToTensor(),
            transforms.Resize((224, 224))
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # transforms.Normalize(mean, std),
        ]
    )

    data_list = []
    for img in images:
        lbl = torch.zeros(40, 1)
        data_list.append([transform(img), lbl])

    predict_loader = DataLoader(dataset=data_list, batch_size=1, shuffle=False)
    prediction = trainer.predict(classifier, predict_loader)  # without fine-tuning
    all = []
    for idx, data_input in enumerate(prediction):
        pred = data_input[2][0]
        all.append(pred)
    return all


def plot_image_ara(img_ara, folder=None, title=None):
    rows = img_ara.shape[0]
    cols = img_ara.shape[1]

    print(rows, cols)

    f, axarr = plt.subplots(rows, cols, figsize=(cols, rows), squeeze=False)
    for c in range(cols):

        for r in range(rows):
            axarr[r, c].get_xaxis().set_ticks([])
            axarr[r, c].get_yaxis().set_ticks([])

            img = img_ara[r][c]
            # img= np.transpose(img, (1,2,0))
            axarr[r, c].imshow(img)

        f.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    if folder == None:
        plt.show()
    else:
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f'{folder}/{title}.png', bbox_inches='tight')

    plt.close()


def get_classifier(attributes, IMAGE_SIZE=128):
    #### load classifier
    config = Parameters()

    attr_dict = attributes

    checkpoint = torch.load(config.inference_param.ckpt_path)
    model = Classification(config.inference_param, attr_dict)
    model.load_state_dict(checkpoint["state_dict"])
    print('Classifier loaded')
    trainer = Trainer(devices=config.hparams.gpu, limit_train_batches=0, limit_val_batches=0)

    return model, trainer


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)