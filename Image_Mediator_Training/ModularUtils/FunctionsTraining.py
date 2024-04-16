import copy
import os
import pickle

# import numpy as np
import torch
import torch
import torchvision
import torchvision.transforms as transforms
import pickle

from matplotlib import pyplot as plt

def top_sort_list(lst, top_list):
    # print("getting latent confounders and observed variables.")
    new_lst = []
    for nd in top_list:
        if nd in lst:
            new_lst.append(nd)

    return new_lst

def get_training_variables(Exp, cur_mechs, interv_no, intv_key):
    all_compare_Var = []
    for mech in cur_mechs:
        # if interv_no not in Exp.train_mech_dict[mech]:
        #     continue
        ret = [lb for lb in Exp.train_mech_dict[mech][interv_no]["compare"] if lb not in all_compare_Var]
        all_compare_Var += ret

    # parents of all joint variables are suitable for intervention.
    intervened_Var = list(intv_key.keys())
    for mech in all_compare_Var:
        intervened_Var += [lb for lb in Exp.train_mech_dict[mech][interv_no]["parents"] if
                    # lb not in intervened_Var + all_compare_Var+Exp.image_labels]
                           lb not in intervened_Var + all_compare_Var]  #should not exclude images from intervening

    compare_Var =[] # [lb for lb in all_compare_Var ]
    for lb in all_compare_Var:
        # if lb in Exp.image_labels:
        #     compare_Var.append("R"+lb)
        # else:
        if lb not in Exp.image_labels+Exp.rep_labels:
            compare_Var.append(lb)

    real_data_vars=[]
    for lb in intervened_Var+ compare_Var:
        if lb not in real_data_vars+Exp.image_labels: # these variables are taken from batchxdim dataset. we have a different batchx3ximage_dimximage_dim dataset
            real_data_vars+=[lb]

    base= Exp.label_names
    return top_sort_list(all_compare_Var, base), top_sort_list(compare_Var, base), top_sort_list(intervened_Var, base), top_sort_list(real_data_vars, base)
    #HnUAn+ U images, HnUAn+\{images,RI}, pa(HnUAn+\RI),  HnUAn+ U pa(HnUAn+)
    #all without any RI.

def image_gradient_penalty(critic, real_image, fake_image, device):

    img_batch_size,C,H,W = real_image.shape
    img_epsilon = torch.rand((img_batch_size,1,1,1)).repeat(1,C,H,W).to(device)  #you repeat or not, it will be
    interpolated_image = img_epsilon* real_image + (1-img_epsilon) * fake_image

    # critic score
    mixed_score = critic(interpolated_image)  # is it okay


    interpolated= interpolated_image

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs= mixed_score,
        grad_outputs=torch.ones_like(mixed_score),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)   #L2 norm
    grad_penalty = torch.mean((gradient_norm-1)**2)
    return grad_penalty

def labels_image_gradient_penalty(critic, real_image, real_labels, fake_image, fake_labels,  isClassifier, device):


    img_batch_size,C,H,W = real_image.shape
    img_epsilon = torch.rand((img_batch_size,1,1,1)).repeat(1,C,H,W).to(device)  #you repeat or not, it will be
    interpolated_image = img_epsilon* real_image + (1-img_epsilon) * fake_image

    label_batch_size,C,H,W = real_labels.shape
    lb_epsilon = torch.rand((label_batch_size,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated_labels = lb_epsilon* real_labels + (1-lb_epsilon) * fake_labels
    # using different random epsilons will it be a problem?

    # critic score
    mixed_score = critic(interpolated_image, interpolated_labels)  # is it okay


    if isClassifier:
        interpolated= interpolated_labels
    else:
        interpolated= interpolated_image

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs= mixed_score,
        grad_outputs=torch.ones_like(mixed_score),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)   #L2 norm
    grad_penalty = torch.mean((gradient_norm-1)**2)
    return grad_penalty



def calc_gradient_penalty(critic, real_data, fake_data, device='cpu', pac=10):
    """Compute the gradient penalty."""
    alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
    alpha = alpha.repeat(1, pac, real_data.size(1))
    alpha = alpha.view(-1, real_data.size(1))

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    disc_interpolates = critic(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
    gradient_penalty = ((gradients_view) ** 2).mean()
    # gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

    return gradient_penalty


def save_results(Exp, saved_path,  tvd_diff, kl_diff):


    for dist in tvd_diff:
        os.makedirs(saved_path + "/tvd", exist_ok=True)
        os.makedirs(saved_path + "/kl", exist_ok=True)
        torch.save(torch.FloatTensor(tvd_diff[dist]), saved_path + "/tvd/"+dist)
        torch.save(torch.FloatTensor(kl_diff[dist]), saved_path + "/kl/"+dist)


    # for key in all_generated_labels:
    #     torch.save(all_generated_labels[key], saved_path + f"/{str(key)}generated_labels")
    #     torch.save(torch.FloatTensor(G_avg_losses), saved_path + f"/{str(key)}G_avg_losses")
    #
    #     if key in all_real_labels:
    #         torch.save(all_real_labels[key], saved_path + f"/{str(key)}real_labels")
    #     torch.save(torch.FloatTensor(D_avg_losses), saved_path + f"/{str(key)}D_avg_losses")


def save_checkpoint(Exp , saved_path, label_generators, G_optimizers, label_discriminators, D_optimizers):
    print("=> Saving checkpoint")
    gen_checkpoint = {}
    for label in label_generators:
        gen_checkpoint[label + "state_dict"] = label_generators[label].state_dict()
        gen_checkpoint[label + "optimizer"] = G_optimizers[label].state_dict()

    os.makedirs(saved_path + f"/gen_checkpoints", exist_ok=True)
    gfile = saved_path + f"/gen_checkpoints/epoch{Exp.curr_epoochs:03}.pth"
    last_gfile = saved_path + f"/gen_checkpoints/epochLast.pth"
    torch.save(gen_checkpoint, gfile)
    torch.save(gen_checkpoint, last_gfile)


    #--------
    disc_checkpoint = {}
    for name in label_discriminators:
        disc_checkpoint[name + "state_dict"] = label_discriminators[name].state_dict()
        disc_checkpoint[name + "optimizer"]= D_optimizers[name].state_dict()

    os.makedirs(saved_path + f"/disc_checkpoints", exist_ok=True)
    dfile = saved_path + f"/disc_checkpoints/epoch{Exp.curr_epoochs:03}.pth"
    last_dfile = saved_path + f"/disc_checkpoints/epochLast.pth"
    torch.save(disc_checkpoint, dfile)
    torch.save(disc_checkpoint, last_dfile)



def load_checkpointed_generators(saved_path, label_generators, G_optimizers, lr):
    gfile = saved_path + "/checkpoints_generators.pth"
    checkpoint = torch.load(gfile, map_location="cuda")

    for label in label_generators:
        label_generators[label].load_state_dict(checkpoint[label + "state_dict"])

    # print("generator Load", checkpoint)

    for i, label in enumerate(G_optimizers):
        G_optimizers[label].load_state_dict(checkpoint["optimizer" + str(i)])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in G_optimizers[label].param_groups:
            param_group["lr"] = lr


def load_checkpointed_discriminators(saved_path, label_discriminators, D_optimizers, lr):
    dfile = saved_path + "/checkpoints_discriminator.pth"
    checkpoint = torch.load(dfile, map_location="cuda")

    for id in range(len(label_discriminators)):
        label_discriminators[id].load_state_dict(checkpoint["dstate_dict" + str(id)])

    for id, optimizer in enumerate(D_optimizers):
        optimizer.load_state_dict(checkpoint["doptimizer" + str(id)])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def load_checkpoint(saved_path, label_generators, G_optimizers, label_discriminators, D_optimizers, lr):
    print("=> Loading checkpoint")

    load_checkpointed_generators(saved_path, label_generators, G_optimizers, lr)
    load_checkpointed_discriminators(saved_path, label_discriminators, D_optimizers, lr)


def plot_to_tensorboard(
        writer, loss_critic, loss_gen, real, fake, tensorboard_step
):
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)

    with torch.no_grad():
        # take out (up to) 8 examples to plot
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)


def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    print("validating state dicts")
    if len(model_state_dict_1) != len(model_state_dict_2):
        print(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1:]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1:]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
            model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            print(f"Tensor mismatch: {v_1} vs {v_2}")
            return False

    print("No mismatch")
