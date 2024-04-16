
from matplotlib import pyplot as plt
import os
import glob
from zipfile import ZipFile
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from PIL import Image



class Celeba(Dataset):
    def __init__(self, root=None, dataset_name=None, label_name=None):

        if dataset_name == "img_align_celeba":            #/local/scratch/a/rahman89/CelebA, img_align_celeba, list_attr_celeba
            self.data_path = sorted(glob.glob(f'{root}/{dataset_name}/img_align_celeba/*.jpg'))
            label_path = f"{root}/{label_name}.csv"
            sep = ","
            rowst = 1
            colst = 1

        if dataset_name == "CelebAMask-HQ":   #/local/scratch/a/rahman89/CelebAMask-HQ, CelebA-HQ-img, CelebAMask-HQ-attribute-anno
            self.data_path = sorted(glob.glob(f'{root}/{dataset_name}/CelebA-HQ-img/*.jpg'))
            label_path = f"{root}/{dataset_name}/{label_name}.txt"
            sep = " "
            rowst = 2
            colst = 2

        label_list = open(label_path).readlines()[rowst:]

        self.attributes = open(label_path).readlines()[1].split(' ')


        data_label = []
        for i in range(len(label_list)):
            each_row = label_list[i].split(sep)
            # if i<10:
            #     print(label_list[i], each_row)
            data_label.append(each_row)

        # transform label into 0 and 1
        for m in range(len(data_label)):
            data_label[m] = [n.replace('-1', '0') for n in data_label[m]][colst:]  # excluding the image id
            data_label[m] = [int(p) for p in data_label[m]]

        self.label_values = data_label

        # Data transforms
        self.transform = transforms.Compose(
            [transforms.Resize((224,224)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        image_set = Image.open(self.data_path[idx])
        image_tensor = self.transform(image_set)
        image_label = torch.Tensor(self.label_values[idx])

        return image_tensor, image_label


indices = list(range(202599))







celebadata = Celeba('/local/scratch/a/rahman89/CelebA', 'img_align_celeba', 'list_attr_celeba')
var = dataset[0]
