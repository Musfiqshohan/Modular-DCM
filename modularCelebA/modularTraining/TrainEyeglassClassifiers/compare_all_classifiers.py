
import pandas as pd
import torch
from modularTraining.constantFunctions import get_train_test_loaders, get_idata, plot_image_ara, \
    get_one_attribute_loader
from modularTraining.constantFunctions import get_sexWithAge

from Classifiers.mobileNet.model import MobileNet
from torch import nn
from Classifiers.mobileNet.train_valid_test import test
import modularTraining.constant_paths as const
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
domain_dataset2, trainloader2, validloader2, testloader2 = get_train_test_loaders(targetAtt, lb_file, img_file)


m0_testloader2= get_one_attribute_loader(domain_dataset2, att1='Male', val1=0, att2='Eyeglasses')
m1_testloader2= get_one_attribute_loader(domain_dataset2, att1='Male', val1=1, att2='Eyeglasses')

e0_testloader2= get_one_attribute_loader(domain_dataset2, att1='Eyeglasses', val1=0, att2='Eyeglasses')
e1_testloader2= get_one_attribute_loader(domain_dataset2, att1='Eyeglasses', val1=1, att2='Eyeglasses')


m0e0_testloader2 = get_sexWithAge(domain_dataset2, att1='Male', val1=0, att2='Eyeglasses', val2=0)
m1e0_testloader2 = get_sexWithAge(domain_dataset2, att1='Male', val1=1, att2='Eyeglasses', val2=0)
m0e1_testloader2 = get_sexWithAge(domain_dataset2, att1='Male', val1=0, att2='Eyeglasses', val2=1)
m1e1_testloader2 = get_sexWithAge(domain_dataset2, att1='Male', val1=1, att2='Eyeglasses', val2=1)





columns= ['Classifier', 'Sex=0', 'Sex=1', 'Eyeglasses=0', 'Eyeglasses=1', 'Sex = 0,Eyeglass = 0',  'Sex = 0,Eyeglass = 1', 'Sex = 1,Eyeglass = 0', 'Sex = 1,Eyeglass = 1']


test_images=[]
for img,lb in m0e1_testloader2:
    test_images.append(img.cpu().permute(0, 2, 3, 1))
test_images= torch.cat(test_images)
plot_image_ara(test_images[0:20].unsqueeze(0))

test_images=[]
for img,lb in m1e1_testloader2:
    test_images.append(img.cpu().permute(0, 2, 3, 1))
test_images= torch.cat(test_images)
plot_image_ara(test_images[0:20].unsqueeze(0))


num_labels = 1
save_path = "/{const.project_root}/modularCelebA/Classifiers/new_weights"
learning_rate = 1e-3

domain_list = ['dom1', 'intv', 'intv_aug']
names = ['trained on domain1 dataset',
         'trained on intv dataset',
         'Augmented'
         ]


accuracy={}
for iter, cur_domain in enumerate(domain_list):
    cur_checkpoint_path = f'{save_path}/{cur_domain}_model_checkpoint.pth'
    checkpoint_path = cur_checkpoint_path
    model = MobileNet(num_labels).to('cuda')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # this is very important

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print('--->', names[iter])



    a1= test(model, [], ['Eyeglasses'], m0_testloader2, criterion, num_labels=num_labels)
    a2= test(model, [], ['Eyeglasses'], m1_testloader2, criterion, num_labels=num_labels)
    a3= test(model, [], ['Eyeglasses'], e0_testloader2, criterion, num_labels=num_labels)
    a4= test(model, [], ['Eyeglasses'], e1_testloader2, criterion, num_labels=num_labels)


    a5= test(model, [], ['Eyeglasses'], m0e0_testloader2, criterion, num_labels=num_labels)
    a6= test(model, [], ['Eyeglasses'], m0e1_testloader2, criterion, num_labels=num_labels)
    a7= test(model, [], ['Eyeglasses'], m1e0_testloader2, criterion, num_labels=num_labels)
    a8= test(model, [], ['Eyeglasses'], m1e1_testloader2, criterion, num_labels=num_labels)



    #
    accuracy[cur_domain]=[a1[0], a2[0], a3[0], a4[0], a5[0], a6[0], a7[0], a8[0]]



df = pd.DataFrame(columns=columns)
for dom in accuracy:
    print(dom, accuracy[dom])

    for iter in range(len(accuracy[dom])):
        accuracy[dom][iter]= round(accuracy[dom][iter],3)

    cur_row= [dom]+accuracy[dom]
    df = pd.concat([pd.DataFrame([cur_row], columns=df.columns), df], ignore_index=True)


print(df.to_string())

# headers = ['Epoch', 25, 150, 300]
# print(tabulate(table))