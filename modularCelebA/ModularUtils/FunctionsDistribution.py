import numpy as np

from ModularUtils.ControllerConstants import generate_permutations

###################  Distribution comparison ###############


def compare_conditionals_within(Exp, dataset, observed_var, conditioning_var, names):

    dist_dict = {}

    dims_list1 = [Exp.label_dim[lb] for lb in observed_var]
    Yperms = generate_permutations(dims_list1)

    dims_list2 = [Exp.label_dim[lb] for lb in conditioning_var]
    X_perms = generate_permutations(dims_list2)

    for xp in X_perms:
        Xdict = dict(zip(conditioning_var, xp))
        for yp in Yperms:
            Ydict = dict(zip(observed_var, yp))
            YXdict={**Ydict, **Xdict}
            dist_dict[tuple(YXdict.values())]= conditional_prob(dataset, names, Ydict, Xdict)

    # print("distribution", dist_dict)

    return dist_dict


def conditional_prob(data, names, Y,X):

    # all ={**Y, **X}
    # indices = [ControllerConstants.label_names.index(lb) for lb in all]

    y_ind = [names.index(lb) for lb in Y]
    x_ind = [names.index(lb) for lb in X]

    X_values = np.array(list(X.values())).transpose()
    Y_values = np.array(list(Y.values())).transpose()

    # chosen = data[:, indices].numpy().astype(int)

    # values = np.array(list(X.values())).transpose()
    iterations = len(list(X.values()))
    save = []
    # for r in range(X_values.shape[0]):

        # c1= data[:,x_ind].numpy().astype(int)
        # c2 = X_values[r]
        # check = np.all(c1 == c2,
        #                axis=1)  # Test whether all array elements along a given axis evaluate to True
    chosen_X= data[:,x_ind]
    cond_idx = np.where(np.all(chosen_X == X_values, axis=1))

    conditioned= data[cond_idx]
    chosen_Y = conditioned[:,y_ind]
    final = np.where(np.all(chosen_Y == Y_values, axis=1))

    # cond_prb = (len(final[0])+ 10 ** -6)/(conditioned.shape[0]+ 10 ** -6)   #why division by zero occurs
    cond_prb = (len(final[0]))/(conditioned.shape[0]+ 10 ** -6)   # cant add 10 ** -6 in the numerator, cz then even if no occurrence, num/den becomes 1

    save.append(cond_prb)


    # ret= np.asarray(save)

    return save[0]


def get_joint_distributions_from_samples(Exp, observed_var, corrensponding_samples):
    dim_list = [Exp.label_dim[lb] for lb in observed_var]
    observe_perms = generate_permutations(dim_list)

    combinations,  count = np.unique(corrensponding_samples, axis=0, return_counts = True)

    upd_dist = {}
    for comb in observe_perms:
        upd_dist[tuple(list(comb))] = 1e-6

    total =corrensponding_samples.shape[0]
    for comb,cnt in zip(combinations,count):
        upd_dist[tuple(list(comb))] =  cnt/total


    return upd_dist




def calculate_TVD(dist1, dist2, doPrint):


    if len(dist1) != len(dist2):
        # raise ValueError('distribution doesnt match size')
        return 10000
    tvd =0
    for perm in dist1:
        tvd += abs(dist1[perm] - dist2[perm])
        r1 = round(dist1[perm], 3)
        r2 = round(dist2[perm], 3)

        r3 = abs(dist1[perm] - dist2[perm])
        if doPrint == True and r3 > 0.01:
            print("perm:", perm, "tvd", r3)
    return tvd*0.5


def calculate_KL(gen, real, doPrint):

    if len(real) != len(gen):
        raise ValueError('distribution doesnt match size')

    kl =0
    for perm in real:
        if real[perm]==0 or gen[perm]==0:
            continue
        kl += (real[perm])* np.log(real[perm]/(gen[perm]))
        r1 = round(real[perm], 3)
        r2 = round(gen[perm], 3)

        # r3 = real[perm]* np.log(real[perm]/(gen[perm]+1e-6))
        # if doPrint == True and r3 > 0.01:
        #     print("perm:", perm, "tvd", r3)

    # kl_pq = rel_entr(list(real.values()), list(gen.values()))
    # print('KL(P || Q): %.3f nats' % sum(kl_pq))

    # print("->", kl)
    return kl


def match_with_true_dist(Exp, observed_var, samples, distribution, feature, doPrint):

    upd_dist= get_joint_distributions_from_samples(Exp, observed_var, samples, feature)

    if doPrint:
        print("True", distribution)
        ll= min(len(distribution), 5)
        print(sorted(distribution.items(), key=lambda item: -item[1])[0:ll])
        print("Fake", upd_dist)
        print(sorted(upd_dist.items(), key=lambda item: -item[1])[0:ll])

    tvd= calculate_TVD(distribution, upd_dist, doPrint)
    # tvd= 1000
    kl = calculate_KL(upd_dist, distribution, doPrint)

    return tvd, kl ,  distribution, upd_dist


