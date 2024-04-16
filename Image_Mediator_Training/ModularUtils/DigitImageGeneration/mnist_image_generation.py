import copy
import os
import pickle
import random
import torch
import numpy as np
import random
import torchvision.transforms as transforms
from numpy import uint8
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from skimage.transform import resize

from ModularUtils.DigitImageGeneration.ColoringMnist import color_grayscale_arr
from ModularUtils.DigitImageGeneration.morphomnist import perturb, morpho, io


def plot_trained_digits(rows, columns, images, title, SAVED_PATH):
    # fig = plt.figure(figsize=(13, 1))
    fig = plt.figure(figsize=(13, 8))
    # columns = 6
    # rows = 3
    # ax enables access to manipulate each of subplots
    ax = []

    for i in range(columns * rows):
        img = images[i]
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i + 1))
        # ax[-1].set_title("Label: " + str(label))  # set title
        ax[-1].set_title(title[i], fontsize=20)  # set title
        ax[-1].set_title(title[i])  # set title

        plt.xticks([])
        plt.yticks([])

        plt.imshow(img)

    if SAVED_PATH==None:
        plt.show()
    else:
        os.makedirs(SAVED_PATH, exist_ok=True)
        plt.savefig(f'{SAVED_PATH}/{title}.png', bbox_inches='tight')


def plot_dataset_digits(image):
    fig = plt.figure(figsize=(13, 8))
    columns = 1
    rows = 1
    # ax enables access to manipulate each of subplots
    ax = []

    for i in range(columns * rows):
        img, label = image
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("Label: " + str(label))  # set title

        plt.imshow(img)

    plt.show()


# #
# perturbations = (
#     lambda m: m.binary_image,  # No perturbation
#     perturb.Thinning(amount=.7),
#     perturb.Thickening(amount=1.),
#     perturb.Swelling(strength=3, radius=7),
#     perturb.Fracture(num_frac=3)
# )
# with open("input_dir/test.txt") as f:
#     data = f.read()
#     print(data)
#
#
# images = io.load_idx("input_dir/t10k-images-idx3-ubyte.gz")
# perturbed_images = np.empty_like(images)
# print(perturbations)
# perturbation_labels = np.random.randint(len(perturbations), size=len(images))
# print(perturbation_labels)
# # for n in range(len(images)):
# for n in range(10):
#     morphology = morpho.ImageMorphology(images[n], scale=4)
#     perturbation = perturbations[perturbation_labels[n]]
#     perturbed_hires_image = perturbation(morphology)
#     perturbed_images[n] = morphology.downscale(perturbed_hires_image)
#     colored_arr = color_grayscale_arr(perturbed_images[n], color="blue")
#
#     print(type(perturbed_images[n]))
#     plt.imshow(perturbed_images[n])
#     plt.show()
#     print(type(colored_arr))
#     plt.imshow(colored_arr)
#     plt.show()
#     break


# io.save_idx(perturbed_images, "output_dir/images-idx3-ubyte.gz")
# io.save_idx(perturbation_labels, "output_dir/pert-idx1-ubyte.gz")


def label_to_digit_image(Exp, images_data, labels_data, digit, color_id, thickness, colorset):
    dig_indices = {}
    for dig in range(10):
        dig_indices[dig] = []
    for id, dig in enumerate(labels_data):
        dig_indices[dig].append(id)

    dig_image_id = random.sample(dig_indices[digit], 1)[0]

    # print("Before perturbation")
    # area, length, rlthickness, slant, width, height = measure_image(images_data[dig_image_id])

    perturbations = (
        # lambda m: m.binary_image,  # No perturbation
        perturb.Thinning(amount=0.6),
        perturb.Thickening(amount=0.3),
    )

    # thickness= thickness+1
    change_shape = perturbations[thickness]

    resized_image = resize(images_data[dig_image_id], (Exp.IMAGE_SIZE, Exp.IMAGE_SIZE))

    morphology = morpho.ImageMorphology(resized_image, scale=4)
    perturbed_hires_image = change_shape(morphology)
    perturbed_images = morphology.downscale(perturbed_hires_image)

    colored_arr = color_grayscale_arr(perturbed_images, color=colorset[color_id])


    # change_shape = perturbations[2]
    # morphology = morpho.ImageMorphology(images_data[dig_image_id], scale=4)
    # perturbed_hires_image = change_shape(morphology)
    # perturbed_images = morphology.downscale(perturbed_hires_image)
    # colored_arr = color_grayscale_arr(perturbed_images, color=["red", "green", "blue"][color_id])
    # plt.imshow(colored_arr)
    # plt.show()


    return colored_arr

#data with digit, color, thickness
def produce_result_image(Exp, input_dir, result_dataset, intv_no, num_images, colorset, SAVE_DATASET, Load_prev):
    # result_dataset = []
    # for label in ["Ydigit1", "Ydigit2", "Ycolor", "Ythick"]:
    #     file_name = Exp.file_roots[intv_no] + label  + ".pkl"
    #     with open(file_name, 'rb') as fp:
    #         label_data = pickle.load(fp)
    #     label_data = torch.FloatTensor(label_data)
    #     label_size = len(label_data)
    #     result_dataset.append(label_data.view(label_size, 1))

    # result_dataset = torch.cat(result_dataset, 1).to(Exp.DEVICE)
    print(result_dataset.shape)

    # images_data = io.load_idx(
    #     "/local/scratch/a/rahman89/PycharmProjects/conditional-DCGAN/CausalMNISTAddition/input_dir/train-images-idx3-ubyte.gz")
    # labels_data = io.load_idx(
    #     "/local/scratch/a/rahman89/PycharmProjects/conditional-DCGAN/CausalMNISTAddition/input_dir/train-labels-idx1-ubyte.gz")

    images_data = io.load_idx(input_dir)
    labels_data = io.load_idx(input_dir)


    if Load_prev==True:
        perturbed_images_digit1 = io.load_idx(f"{Exp.file_roots + str(intv_no)}Ydigit1images.gz")
        perturbing_labels_digit1 = io.load_idx(f"{Exp.file_roots + str(intv_no)}Ydigit1labels.gz")
    else:
        perturbed_images_digit1 = np.zeros((num_images, Exp.IMAGE_SIZE, Exp.IMAGE_SIZE, 3))
        # perturbed_images_digit2 = np.zeros((num_images, Exp.IMAGE_SIZE, Exp.IMAGE_SIZE, 3))
        perturbing_labels_digit1 = np.zeros((num_images, 3))
        # perturbing_labels_digit2 = np.zeros((num_images, 3))


    for iter in range(num_images):

        sample = result_dataset[iter, :].detach().cpu().numpy().astype(int)
        print("sample no", iter, sample)

        digit1, color1, thick1 = sample[0], sample[1], sample[2],    ##digit1 with some properties.
        print(digit1, color1, thick1)
        perturbed_images_digit1[iter] = label_to_digit_image(Exp, images_data, labels_data, digit=digit1, color_id=color1, thickness=thick1, colorset=colorset)
        perturbing_labels_digit1[iter] = [digit1, color1, thick1]

        # digit2color= random.randint(0, 2)
        # digit2, color2, thick2 = sample[1], digit2color, 1-sample[3],    #digit2 with opposite properties. color dim= 3, thick dim=2,
        # print(digit2, color2, thick2)
        # perturbed_images_digit2[iter] = label_to_digit_image(iter, images_data, labels_data, digit=digit2, color_id=color2, thickness=thick2)
        # perturbing_labels_digit2[iter] = [digit2, color2, thick2]

        # break
        if iter%1000==4:
            imgg = perturbed_images_digit1[iter]
            plot_dataset_digits((imgg, perturbing_labels_digit1[iter]))
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.ToTensor(),
                                            ])  # ToTensor is needed for tansh

            digit_images = torch.squeeze(transform(imgg.astype(uint8)), dim=0).to(Exp.DEVICE)
            imggg1 = digit_images.permute(1, 2, 0).detach().cpu().numpy()
            plt.imshow(imggg1)
            plt.show()

            if SAVE_DATASET:  # Ydigit1image
                io.save_idx(perturbed_images_digit1, Exp.file_roots + str(intv_no) + "Ydigit1images.gz")
                io.save_idx(perturbing_labels_digit1, Exp.file_roots + str(intv_no) + "Ydigit1labels.gz")
                print("Saved")


    if SAVE_DATASET: #Ydigit1image
        io.save_idx(perturbed_images_digit1, Exp.file_roots+ str(intv_no) + "Ydigit1images.gz")
        io.save_idx(perturbing_labels_digit1, Exp.file_roots+ str(intv_no) + "Ydigit1labels.gz")

    # if Exp.image_labels[1] in SAVE_DATASET:
    #     io.save_idx(perturbed_images_digit2, Exp.file_roots[intv_no] + Exp.image_labels[0]+".gz")
    #     io.save_idx(perturbing_labels_digit2, Exp.file_roots[intv_no] + Exp.image_labels[0]+"labels.gz")


def test_result_data(Exp, intv_no):
    print("TESTING")
    loaded_images = io.load_idx(
        f"{Exp.file_roots+ str(intv_no)}Ydigit1images.gz")
    labels_data = io.load_idx(
        f"{Exp.file_roots+ str(intv_no)}Ydigit1labels.gz")

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    ])

    digit_images = [torch.unsqueeze(transform(img), dim=0).to(Exp.DEVICE) for img in loaded_images]
    digit_images = torch.cat(digit_images, 0)


    for id, img in enumerate(digit_images[0:10]):
        # print(np.min(img), np.max(img))
        print(labels_data[id])
        imggg1 = img.permute(1, 2, 0).detach().cpu().numpy()
        fig, ax = plt.subplots()
        plt.imshow(imggg1)
        plt.show()



def check_any_digit(images_data, labels_data):
    dig_indices = {}
    for dig in range(10):
        dig_indices[dig] = []
    for id, dig in enumerate(labels_data):
        dig_indices[dig[0]].append(id)

    digit = 7
    for id in range(len(dig_indices[digit])):
        imgid = dig_indices[digit][id]
        im = Image.fromarray(images_data[imgid]).convert('RGB')
        img_filename = f"/local/scratch/a/rahman89/PycharmProjects/conditional-DCGAN/SAVED_EXPERIMENTS" \
                       f"/mnist_addition_graph/preprocessed_dataset/result_images/{str(digit)}_{id}.jpeg"
        im.save(img_filename)

        print(f"{id} image of {digit} is saved")
    print("done")



def produce_uniform_images(Exp, intv_no, digits, num_images, SAVE_DATASET):  #used by image mediator


    images_data = io.load_idx(
        "/local/scratch/a/rahman89/PycharmProjects/conditional-DCGAN/CausalMNISTAddition/input_dir/train-images-idx3-ubyte.gz")
    labels_data = io.load_idx(
        "/local/scratch/a/rahman89/PycharmProjects/conditional-DCGAN/CausalMNISTAddition/input_dir/train-labels-idx1-ubyte.gz")

    # num_images = result_dataset.shape[0]
    # perturbed_images_digit = copy.deepcopy(io.load_idx(Exp.file_roots[-1] + "digitimages.gz"))
    # perturbing_labels_digit = copy.deepcopy(io.load_idx(Exp.file_roots[-1] + "digitlabels.gz"))
    perturbed_images_digit = np.zeros((num_images, Exp.IMAGE_SIZE, Exp.IMAGE_SIZE, 3))
    perturbing_labels_digit = np.zeros((num_images, 3))


    colors= torch.randint(0, 3, (num_images,1)).to(Exp.DEVICE)
    thickness= torch.randint(0, 2, (num_images,1)).to(Exp.DEVICE)

    result_dataset= torch.cat([digits, colors, thickness], 1)


    for iter in range(num_images):
        sample = result_dataset[iter, :].detach().cpu().numpy().astype(int)
        print(f"sample no {iter},  image:{sample}")


        digit1, color1, thick1 = sample[0], sample[1], sample[2]
        # print(digit1, color1, thick1)
        perturbed_images_digit[iter] = label_to_digit_image(Exp, images_data, labels_data, digit=digit1, color_id=color1, thickness=thick1)
        perturbing_labels_digit[iter] = [digit1, color1, thick1]


        if iter % 1000 <= 2:
            imgg = perturbed_images_digit[iter]
            plot_dataset_digits((imgg, perturbing_labels_digit[iter]))
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.ToTensor(),
                                            ])  # ToTensor is needed for tansh

            digit_images = torch.squeeze(transform(imgg.astype(uint8)), dim=0).to(Exp.DEVICE)
            imggg1 = digit_images.permute(1, 2, 0).detach().cpu().numpy()
            plt.imshow(imggg1)
            plt.show()


            if SAVE_DATASET:
                io.save_idx(perturbed_images_digit, Exp.file_roots+"intv"+str(intv_no) + "digitimages.gz")
                io.save_idx(perturbing_labels_digit, Exp.file_roots+"intv"+str(intv_no) + "digitlabels.gz")
                print("image saved at", Exp.file_roots+"intv"+str(intv_no)+ "digitimages.gz")
                print("labels saved at", Exp.file_roots+"intv"+str(intv_no) + "digitlabels.gz")


    if SAVE_DATASET:
        io.save_idx(perturbed_images_digit, Exp.file_roots+"intv"+str(intv_no) + "digitimages.gz")
        io.save_idx(perturbing_labels_digit, Exp.file_roots+"intv"+str(intv_no) + "digitlabels.gz")
        print("image saved at", Exp.file_roots+"intv"+str(intv_no) + "digitimages.gz")
        print("labels saved at", Exp.file_roots+"intv"+str(intv_no) + "digitlabels.gz")

    return perturbed_images_digit
    # if Exp.image_labels[1] in SAVE_DATASET:
    #     io.save_idx(perturbed_images_digit2, Exp.file_roots[intv_no] + Exp.image_labels[0]+".gz")
    #     io.save_idx(perturbing_labels_digit2, Exp.file_roots[intv_no] + Exp.image_labels[0]+"labels.gz")
