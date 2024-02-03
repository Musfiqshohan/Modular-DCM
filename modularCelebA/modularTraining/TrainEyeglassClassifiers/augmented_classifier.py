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
from modularTraining.constantFunctions import get_train_test_loaders, get_idata
from modularTraining.constantFunctions import get_sexWithAge

from Classifiers.mobileNet.model import MobileNet
from torch import nn
from Classifiers.mobileNet.train_valid_test import test

from modularTraining.constantFunctions import get_dataloaders
from Classifiers.mobileNet.train_valid_test import train, validation
from Classifiers.mobileNet.train_valid_test import train, validation
import modularTraining.constant_paths as const


exp_name = "sex_eyeglass"
targetAtt = 'Eyeglasses'

dom_name = "domain1"
lb_file = f'/{const.project_root}/modularCelebA/{exp_name}/8_attribute_celeba_{dom_name}.pkl'
img_file = f'/{const.project_root}/modularCelebA/{exp_name}/images_celeba_{dom_name}.pkl'
domain_dataset1, trainloader1, validloader1, testloader1 = get_train_test_loaders(targetAtt, lb_file, img_file)


dom_name = "test_domain"
lb_file = f'/{const.project_root}/modularCelebA/{exp_name}/8_attribute_celeba_{dom_name}.pkl'
img_file = f'/{const.project_root}/modularCelebA/{exp_name}/images_celeba_{dom_name}.pkl'
domain_dataset2, trainloader2, validloader2, testloader2= get_train_test_loaders(targetAtt, lb_file, img_file)
m0y0_testloader2 = get_sexWithAge(domain_dataset2, att1='Male', val1=0, att2='Eyeglasses', val2=0)
m1y0_testloader2 = get_sexWithAge(domain_dataset2, att1='Male', val1=1, att2='Eyeglasses', val2=0)
m0y1_testloader2 = get_sexWithAge(domain_dataset2, att1='Male', val1=0, att2='Eyeglasses', val2=1)
m1y1_testloader2 = get_sexWithAge(domain_dataset2, att1='Male', val1=1, att2='Eyeglasses', val2=1)

# exp_name = "sex_eyeglass"
exp_name = "sex_eyeglass_reproduce"  #reproducing the experiment. Loading the new set of generated images from this folder
intv_labels={}
folder= f'/{const.project_root}/modularCelebA/{exp_name}'
male_list=[]
eye_list=[]
img_list=[]
for m,y in zip([0,0,1,1], [0,1,0,1]):
	# ll, ii = get_idata(f'fakeIdom{m}y{y}/images_Idom{m}y{y}.pkl', f'fakeIdom{m}y{y}/labels_Idom{m}y{y}.pkl')  #getting  m0y0 data samples
	ll, ii = get_idata(folder, f'fakeIdom{m}e{y}/images_Idom{m}e{y}.pkl', f'fakeIdom{m}e{y}/labels_Idom{m}e{y}.pkl')  #getting  m0e0 data samples
	male_list.append(ll['Male'])
	eye_list.append(ll['Eyeglasses'])
	img_list.append(ii['I'])

intv_labels['Male'] = np.concatenate(male_list)
intv_labels['Eyeglasses'] = np.concatenate(eye_list)
intv_images =  torch.cat(img_list)

h= intv_images.shape[0]
rnices= torch.randint(0, h, (h,))
intv_labels['Male']= intv_labels['Male'][rnices]
intv_labels['Eyeglasses']= intv_labels['Eyeglasses'][rnices]
intv_images= intv_images[rnices]
intv_age= intv_labels['Eyeglasses']

# creating augmented dataset
dom1_age, dom1_images= domain_dataset1[targetAtt], domain_dataset1['I']
aug_age= torch.cat([dom1_age, torch.tensor(intv_age)])
aug_images= torch.cat([dom1_images, torch.tensor(intv_images)])
randices= torch.randint(0, aug_age.shape[0], (aug_age.shape[0],))
aug_age= aug_age[randices].view(-1,1).float()
aug_images= aug_images[randices]
trainloaderA, validloaderA, testloaderA = get_dataloaders(aug_age, aug_images,split_t=0.90, split_v=0.95, img_size=224)

# CLassifier
train_all_losses2 = []
train_all_acc2 = []
val_all_losses2 = []
val_all_acc2 = []
test_all_losses2 = 0.0
learning_rate = 1e-3
epochs = 30
num_labels = 1
save_path = f"/{const.project_root}/modularCelebA/Classifiers/new_weights"
cur_domain='intv_aug'
cur_checkpoint_path = f'{save_path}/{cur_domain}_model_checkpoint.pth'
print(cur_checkpoint_path)

best_acc = 0.0
# instantiate Net class
mobilenet = MobileNet(num_labels)
# use cuda to train the network
mobilenet.to('cuda')
#loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(mobilenet.parameters(), lr=learning_rate, betas=(0.9, 0.999))

for epoch in range(epochs):
	train(mobilenet,  optimizer, epoch, train_all_losses2, train_all_acc2, trainloaderA, criterion, num_labels=num_labels)
	acc = validation(mobilenet, val_all_losses2, val_all_acc2, best_acc, validloaderA, criterion, num_labels=num_labels)
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