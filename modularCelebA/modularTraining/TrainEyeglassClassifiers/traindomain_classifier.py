import copy

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import copy
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from modularTraining.constantFunctions import get_train_test_loaders
from modularTraining.constantFunctions import get_sexWithAge

from Classifiers.mobileNet.model import MobileNet
from torch import nn
from Classifiers.mobileNet.train_valid_test import test
import modularTraining.constant_paths as const


# male_testloader1, female_testloader1= get_sex_loaders(domain_dataset1)
# m0y0_testloader1 = get_sexWithAge(domain_dataset1, male=0, young=0)
# m0y1_testloader1 = get_sexWithAge(domain_dataset1, male=0, young=1)
# m1y0_testloader1 = get_sexWithAge(domain_dataset1, male=1, young=0)
# m1y1_testloader1 = get_sexWithAge(domain_dataset1, male=1, young=1)

# domain_dataset2, trainloader2, validloader2, testloader2= get_train_test_loaders(dom_name= "test_domain")
# male_testloader2, female_testloader2= get_sex_loaders(domain_dataset2, needSameSize=False)
# get_sexWithAge(domain_dataset, att1, val1, att2, val2 , doNormalize=False)
# m0y0_testloader2 = get_sexWithAge(domain_dataset2, male=0, young=0)
# m1y0_testloader2 = get_sexWithAge(domain_dataset2, male=1, young=0)
# m0y1_testloader2 = get_sexWithAge(domain_dataset2, male=0, young=1)
# m1y1_testloader2 = get_sexWithAge(domain_dataset2, male=1, young=1)

exp_name = "sex_eyeglass"
dom_name = "domain1"
targetAtt = 'Eyeglasses'
lb_file = f'/{const.project_root}/modularCelebA/{exp_name}/8_attribute_celeba_{dom_name}.pkl'
img_file = f'/{const.project_root}/modularCelebA/{exp_name}/images_celeba_{dom_name}.pkl'
domain_dataset1, trainloader1, validloader1, testloader1 = get_train_test_loaders(targetAtt, lb_file, img_file)
m0y0_testloader1 = get_sexWithAge(domain_dataset1, att1='Male', val1=0, att2='Eyeglasses', val2=0)
m1y0_testloader1 = get_sexWithAge(domain_dataset1, att1='Male', val1=1, att2='Eyeglasses', val2=0)
m0y1_testloader1 = get_sexWithAge(domain_dataset1, att1='Male', val1=0, att2='Eyeglasses', val2=1)
m1y1_testloader1 = get_sexWithAge(domain_dataset1, att1='Male', val1=1, att2='Eyeglasses', val2=1)

dom_name = "test_domain"
targetAtt = 'Eyeglasses'
lb_file = f'/{const.project_root}/modularCelebA/{exp_name}/8_attribute_celeba_{dom_name}.pkl'
img_file = f'/{const.project_root}/modularCelebA/{exp_name}/images_celeba_{dom_name}.pkl'
domain_dataset2, trainloader2, validloader2, testloader2 = get_train_test_loaders(targetAtt, lb_file, img_file)
m0y0_testloader2 = get_sexWithAge(domain_dataset2, att1='Male', val1=0, att2='Eyeglasses', val2=0)
m1y0_testloader2 = get_sexWithAge(domain_dataset2, att1='Male', val1=1, att2='Eyeglasses', val2=0)
m0y1_testloader2 = get_sexWithAge(domain_dataset2, att1='Male', val1=0, att2='Eyeglasses', val2=1)
m1y1_testloader2 = get_sexWithAge(domain_dataset2, att1='Male', val1=1, att2='Eyeglasses', val2=1)

train_all_losses2 = []
train_all_acc2 = []
val_all_losses2 = []
val_all_acc2 = []
test_all_losses2 = 0.0
learning_rate = 1e-3
epochs = 20

num_labels = 1

save_path = f"/{const.project_root}/modularCelebA/Classifiers/new_weights"

from Classifiers.mobileNet.train_valid_test import train, validation

cur_domain = 'dom1'
cur_checkpoint_path = f'{save_path}/{cur_domain}_model_checkpoint.pth'

best_acc = 0.0
# instantiate Net class
mobilenet = MobileNet(num_labels)
# use cuda to train the network
mobilenet.to('cuda')
# loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(mobilenet.parameters(), lr=learning_rate, betas=(0.9, 0.999))

for epoch in range(epochs):
    train(mobilenet, optimizer, epoch, train_all_losses2, train_all_acc2, trainloader1, criterion,
          num_labels=num_labels)
    acc = validation(mobilenet, val_all_losses2, val_all_acc2, best_acc, validloader1, criterion, num_labels=num_labels)
    # record the best model
    if acc > best_acc:
        checkpoint_path = cur_checkpoint_path
        best_acc = acc
        # save the model and optimizer
        torch.save({'model_state_dict': mobilenet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
        print('new best model saved')
    print("========================================================================")

mobilenet.eval()
attr_acc = []
test(mobilenet, [], ['Eyeglasses'], m0y0_testloader2, criterion, num_labels=num_labels)
test(mobilenet, [], ['Eyeglasses'], m0y1_testloader2, criterion, num_labels=num_labels)
test(mobilenet, [], ['Eyeglasses'], m1y0_testloader2, criterion, num_labels=num_labels)
test(mobilenet, [], ['Eyeglasses'], m1y1_testloader2, criterion, num_labels=num_labels)
