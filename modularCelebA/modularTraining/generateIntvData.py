# CelebA image generation using Conditional DCGAN
import itertools
import pickle

from math import ceil
from tqdm import tqdm

from torch.utils.data import DataLoader

from ModularUtils.ControllerConstants import map_dictfill_to_discrete, get_multiple_labels_fill
from ModularUtils.Experiment_Class import Experiment
from ModularUtils.FunctionsDistribution import get_joint_distributions_from_samples
from interfaceGan.models.stylegan_generator import StyleGANGenerator
from lightningmodules.classification import Classification

import numpy as np
import torch
from pytorch_lightning import Trainer
import os, os.path as osp
from matplotlib import pyplot as plt

from modularTraining.TrainEyeglassClassifiers.celeba_eye_graph import set_celeba_eye
from modularTraining.constantFunctions import get_transform, Parameters, get_prediction
from modularTraining.modularEyeTrain import get_generators, get_generated_labels
import modularTraining.constant_paths as const


def generatedCorrectly(model, image, trainer, att_val):

    image = image[0]
    transform = get_transform(image_size=IMAGE_SIZE)

    data_list = []
    lbl = torch.zeros(40, 1)
    data_list.append([transform(image), lbl])

    # plt.imshow(data_list[0][0].cpu().permute(1,2,0))
    # plt.show()

    predict_loader = DataLoader(dataset=data_list, batch_size=1, shuffle=False)
    prediction = trainer.predict(model, predict_loader)  # without fine-tuning
    for idx, data_input in enumerate(prediction):
        pred= data_input[2][0]


    att1, att2= att_val.keys()
    a1,a2= att_val.values()
    b1,b2=0,0
    if att1 in pred:
        b1 = 1
    if att2 in pred:
        b2 = 1

    if (a1,a2) == (b1,b2):
        # plt.imshow(data_list[0][0].permute(1,2,0))
        # plt.show()
        # print(True)
        return True


    return False





################################# Generation
def sample_codes(generator, num, latent_space_type='Z', seed=0):
  """Samples latent codes randomly."""
  np.random.seed(seed)
  codes = generator.easy_sample(num)
  if generator.gan_type == 'stylegan' and latent_space_type == 'W':
    codes = torch.from_numpy(codes).type(torch.FloatTensor).to(generator.run_device)
    codes = generator.get_value(generator.model.mapping(codes))
  return codes

def generate_stylegan_images(generator, boundaries,  ATTRS, att_val, num_samples ):
    # @title { display-mode: "form", run: "auto" }

    # num_samples= 5  # @param {type:"slider", min:1, max:8, step:1}
    latent_space_type = 'Z'  # @param ['Z', 'W']
    synthesis_kwargs = {}

    # Female young: -1,-1
    # Female old: -3, 4
    # Male old: 3, 3
    # Male young: 3, -1
    # gender = 3  # @param {type:"slider", min:-3.0, max:3.0, step:0.1}
    # age = -1  # @param {type:"slider", min:-3.0, max:3.0, step:0.1}

    # for iter in range(5):
    noise_seed = torch.randint(0, 100000, (1,))  # @param {type:"slider", min:0, max:1000, step:1}

    latent_codes = sample_codes(generator, num_samples, latent_space_type, noise_seed)
    # images = generator.easy_synthesize(latent_codes, **synthesis_kwargs)['image']   #generating base images

    new_codes = latent_codes.copy()
    for i, attr_name in enumerate(ATTRS):
        # new_codes += boundaries[attr_name] * eval(attr_name)
        new_codes += boundaries[attr_name] * att_val[attr_name]

    new_images = generator.easy_synthesize(new_codes, **synthesis_kwargs)['image']
    # imshow(new_images, col=num_samples)

    return new_images





def plot_image_ara(img_ara, folder=None, title=None):
    rows=img_ara.shape[0]
    cols=img_ara.shape[1]

    print(rows,cols)

    f, axarr = plt.subplots(rows, cols, figsize=(cols, rows), squeeze=False)
    for c in range(cols):

        for r in range(rows):
            axarr[r, c].get_xaxis().set_ticks([])
            axarr[r, c].get_yaxis().set_ticks([])

            img= img_ara[r][c]
            # img= np.transpose(img, (1,2,0))
            axarr[r, c].imshow(img)


        f.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    if folder==None:
        plt.show()
    else:
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f'{folder}/{title}.png', bbox_inches='tight')

    plt.close()



# def do_process(generated_labels):
def do_process(generator, classifier, trainer, male,eye ,combinations, num_samples, exp_name, filename):

    new_images = []
    new_labels = {'Male': [], 'Eyeglasses': []}

    transform= get_transform(image_size=IMAGE_SIZE) #normalizes too


    for comb in combinations:
        for iter in tqdm(range(num_samples)):
            flag = False
            fail_cnt = 0
            while flag == False:
                input_sex, input_age= comb

                # att_val = {'gender': input_sex, 'age': input_age}
                att_val = {'gender': input_sex, 'eyeglasses': input_age}
                image = generate_stylegan_images(generator, boundaries, ATTRS, att_val, num_samples=1)

                att_val = {'Male': male, 'Eyeglasses': eye}
                if generatedCorrectly(classifier, image, trainer, att_val) == True:
                    new_images.append(transform(image.squeeze()).unsqueeze(0))
                    new_labels['Male'].append(male)
                    new_labels['Eyeglasses'].append(eye)

                    flag = True

                else:
                    fail_cnt += 1
                    print(f'Failed: {fail_cnt} times')



    transformed = torch.cat(new_images)
    final_images = {'I': transformed}
    save_folder = f'/{const.project_root}/modularCelebA/{exp_name}/fake{filename}'
    os.makedirs(save_folder,exist_ok=True)

    save_loc = f'{save_folder}/images_{filename}.pkl'
    with open(save_loc, 'wb') as f:
        pickle.dump(final_images, f)

    save_loc = f'{save_folder}/labels_{filename}.pkl'
    with open(save_loc, 'wb') as f:
        pickle.dump(new_labels, f)

    print('Going to save at', save_loc)


def generate_permutations(lst1, lst2):
    sequences=[lst1, lst2]
    # for dim in dim_list:
    #     sequences.append([i for i in range(dim)])

    lst = []
    for p in itertools.product(*sequences):
        lst.append(p)

    np_ara = np.array(lst)
    return np_ara



def get_sexeye_params(male, eye):

    lst0=[-2,-1]
    lst1=[2,3]

    if male==0 and eye==0:  #female no eye
        combs= generate_permutations(lst0, lst0)
        filename = "Idom0e0"
    elif male==0 and eye==1:  #female with eye
        combs= generate_permutations(lst0, lst1)
        filename = "Idom0e1"
    elif male==1 and eye==0:  #male no eye
        combs= generate_permutations(lst1, lst0)
        filename = "Idom1e0"
    elif male==1 and eye==1:  #male with eye
        combs= generate_permutations(lst1, lst1)
        filename = "Idom1e1"

    return combs, filename

if __name__ == '__main__':



    mod_name = 'celeba-eye-train'

    Exp = Experiment(mod_name, set_celeba_eye,
                     Temperature=1,
                     temp_min=0.1,
                     G_hid_dims=[256, 256],
                     D_hid_dims=[256, 256],
                     # IMAGE_FILTERS=[512, 256, 128],
                     IMAGE_FILTERS=[128, 64, 32],
                     CRITIC_ITERATIONS=1,
                     LAMBDA_GP=10,
                     learning_rate=1e-4,
                     batch_size=200,
                     noise_states=64,
                     latent_state=4,
                     ENCODED_DIM=10,
                     Data_intervs=[{}],
                     num_epochs=301,
                     NOISE_DIM=128,
                     CONF_NOISE_DIM=128,
                     new_experiment=True
                     )

    print(Exp.Data_intervs)
    Exp.intv_batch_size = Exp.batch_size
    os.makedirs(Exp.SAVED_PATH, exist_ok=True)

    Exp.load_which_models = {"Male": False, "Eyeglasses": False}
    cur_hnodes = {"H1": ["Male", "Eyeglasses"]}

    Exp.LAMBDA_GP = 10
    sample_size=10000
    label_generators, optimizersMech = get_generators(Exp, Exp.load_which_models, None)

    gfile=f"/{const.project_root}/modularCelebA/modularTraining/TrainEyeglassClassifiers/SAVED_EXPERIMENTS/celeba-eye-train/gen_checkpoints/epochLast.pth"
    checkpoint = torch.load(gfile, map_location="cuda")
    for label in label_generators:
        label_generators[label].load_state_dict(checkpoint[label + "state_dict"])


    zeros= torch.zeros((int(sample_size/2),1)).to('cuda')
    ones=  torch.ones((int(sample_size/2),1)).to('cuda')
    input= torch.cat([zeros,ones])
    real_labels_fill = get_multiple_labels_fill(Exp, input, [2,2], isImage_labels=False)
    intv_key={'Male': real_labels_fill}
    generated_labels_dict = get_generated_labels(Exp, label_generators, dict(intv_key), ["Male", "Eyeglasses"],
                                                 sample_size)
    generated_labels_full = map_dictfill_to_discrete(Exp, generated_labels_dict, ["Male", "Eyeglasses"])
    fake_dist_dict = get_joint_distributions_from_samples(Exp, ["Male", "Eyeglasses"], generated_labels_full)

    combinations,  count = np.unique(generated_labels_full, axis=0, return_counts = True)
    comb_dict={}
    for iter, comb in enumerate(combinations):
        comb_dict[tuple(comb)]= count[iter]


    #### Pre-trained InterfaceGAN
    IMAGE_SIZE = 224

    # exp_name="sex_eyeglass"
    exp_name="sex_eyeglass_reproduce"
    #
    #### load stylegan
    attr1 = 'gender'
    attr2 = 'eyeglasses'

    ATTRS = [attr1, attr2]
    model_name = "stylegan_celebahq"  # @param ['stylegan_celebahq', 'stylegan_ffhq', 'stylegan2_ffhq', 'stylegan3_ffhq']

    latent_space_type = 'Z'  # @param ['Z', 'W']
    if latent_space_type == 'W' and ('2' in model_name or '3' in model_name):
        raise ValueError('Latent space is not available for StyleGAN 2 and 3')

    generator = StyleGANGenerator(model_name)
    boundaries = {}
    root = f"/{const.project_root}/interfacegan/TrainingCelebA/interfacegan"
    if model_name == "stylegan_celebahq":
        boundaries[attr1] = np.load(f'{root}/boundaries/stylegan_celebahq/stylegan_celebahq_{attr1}_boundary.npy')
        boundaries[attr2] = np.load(f'{root}/boundaries/stylegan_celebahq/stylegan_celebahq_{attr2}_boundary.npy')



    #### load classifier
    config = Parameters()
    checkpoint = torch.load(config.inference_param.ckpt_path)
    classifier = Classification(config.inference_param)
    classifier.load_state_dict(checkpoint["state_dict"])
    print('Classifier loaded')
    trainer = Trainer(devices=config.hparams.gpu, limit_train_batches=0, limit_val_batches=0)


    for male, eye in zip([0,1,0,1], [1,1,0,0]):

        num_samples=comb_dict[(male,eye)]
        cur_combs, filename= get_sexeye_params(male, eye)
        print(cur_combs, filename, num_samples)
        # if m0e0 =0,0 then cur_combs are (-2,-2), (-2,-1), (-1,-2), (-1,-1) .
        # same for other combinations. So, if we have to generate n samples for m0e0, we generate n/4 for each above category.
        num_samples= int(ceil(num_samples/cur_combs.shape[0]))
        gen_images = do_process(generator, classifier, trainer, male,eye, cur_combs, num_samples=num_samples, exp_name= exp_name, filename=filename)



