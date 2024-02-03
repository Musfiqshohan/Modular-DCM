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
from modularTraining.constantFunctions import get_train_test_loaders, get_idata, plot_image_ara, denorm
from modularTraining.constantFunctions import get_sexWithAge

from Classifiers.mobileNet.model import MobileNet
from torch import nn
from Classifiers.mobileNet.train_valid_test import test

from modularTraining.constantFunctions import get_dataloaders
from Classifiers.mobileNet.train_valid_test import train, validation
import modularTraining.constant_paths as const



exp_name = "sex_eyeglass"

dom_name = "test_domain"
targetAtt = 'Eyeglasses'
lb_file = f'/{const.project_root}/modularCelebA/{exp_name}/8_attribute_celeba_{dom_name}.pkl'
img_file = f'/{const.project_root}/modularCelebA/{exp_name}/images_celeba_{dom_name}.pkl'

domain_dataset2, trainloader2, validloader2, testloader2= get_train_test_loaders(targetAtt, lb_file, img_file)
m0y0_testloader2 = get_sexWithAge(domain_dataset2, att1='Male', val1=0, att2='Eyeglasses', val2=0)
m1y0_testloader2 = get_sexWithAge(domain_dataset2, att1='Male', val1=1, att2='Eyeglasses', val2=0)
m0y1_testloader2 = get_sexWithAge(domain_dataset2, att1='Male', val1=0, att2='Eyeglasses', val2=1)
m1y1_testloader2 = get_sexWithAge(domain_dataset2, att1='Male', val1=1, att2='Eyeglasses', val2=1)




# for new images
exp_name = "sex_eyeglass_reproduce"
intv_labels={}
folder= f'/{const.project_root}/modularCelebA/{exp_name}'
male_list=[]
eye_list=[]
img_list=[]

plt_img=[]

for m,y in zip([0,0,1,1], [0,1,0,1]):
	# ll, ii = get_idata(f'fakeIdom{m}y{y}/images_Idom{m}y{y}.pkl', f'fakeIdom{m}y{y}/labels_Idom{m}y{y}.pkl')  #getting  m0y0 data samples
	ll, ii = get_idata(folder, f'fakeIdom{m}e{y}/images_Idom{m}e{y}.pkl', f'fakeIdom{m}e{y}/labels_Idom{m}e{y}.pkl')  #getting  m0e0 data samples

	if y==1:
		plt_img.append(denorm(ii['I'][100:116]).permute(0,2,3,1).unsqueeze(0))


	male_list.append(ll['Male'])
	eye_list.append(ll['Eyeglasses'])
	img_list.append(ii['I'])

#
img_ara= torch.cat(plt_img)
plot_image_ara(img_ara)




intv_labels['Male'] = np.concatenate(male_list)
intv_labels['Eyeglasses'] = np.concatenate(eye_list)
intv_images =  torch.cat(img_list)

h= intv_images.shape[0]
rnices= torch.randint(0, h, (h,))
intv_labels['Male']= intv_labels['Male'][rnices]
intv_labels['Eyeglasses']= intv_labels['Eyeglasses'][rnices]
intv_images= intv_images[rnices]

#
intv_age = np.array(intv_labels['Eyeglasses'], dtype='float32').reshape(-1,1)
trainloaderI, validloaderI, testloaderI = get_dataloaders(intv_age, intv_images, split_t=0.90, split_v=0.95, img_size=224)


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
cur_domain='intv'
# cur_checkpoint_path = f'./{cur_domain}_model_checkpoint.pth'
cur_checkpoint_path = f'{save_path}/{cur_domain}_model_checkpoint.pth'

# Use test data for validation

print(cur_checkpoint_path)

best_acc = 0.0
# instantiate Net class
mobilenet = MobileNet(num_labels)
# use cuda to train the network
mobilenet.to('cuda')
#loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(mobilenet.parameters(), lr=learning_rate, betas=(0.9, 0.999))

epochs = 30

for epoch in range(epochs):
	train(mobilenet,  optimizer, epoch, train_all_losses2, train_all_acc2, trainloaderI, criterion, num_labels=num_labels)
	acc = validation(mobilenet, val_all_losses2, val_all_acc2, best_acc, validloaderI, criterion, num_labels=num_labels)
	# record the best model
	if acc > best_acc and epoch>10:
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