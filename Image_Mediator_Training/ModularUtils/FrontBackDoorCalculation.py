import torch
import pandas as  pd
import  numpy as np


#Taken from : http://www.degeneratestate.org/posts/2018/Sep/03/causal-inference-with-python-part-3-frontdoor-adjustment/

def estiamte_ate_frontdoor_direct(Exp, df, x, y, zs):
    """
    Estiamte the ATE of a system from a dataframe of samples `ds`
    using frontdoor adjustment directly on ML estimates of probability.

    E[Y|do(X) = x'] = \sum_{x,y,z} y P[y|x,z] P(z|x') P(X)

    Arguments
    ---------
    df: pandas.DataFrame
    x: str
    y: str
    zs: list[str]

    Returns
    -------
    ATE: float
    """
    zs_unique = [tuple(a) for a in np.unique(df[zs].values, axis=0)]
    y_unique = np.unique(df[y].values, axis=0)

    # P(X)
    p_x = {
        x_: np.mean(df[x] == x_)
        for x_ in range(Exp.label_dim[x])
    }

    # P(Z|X)
    p_z_x = {
        (x_, z_): np.mean(df
                          .loc[lambda df_: df_[x] == x_]
                          [zs]
                          .values == z_)
        for x_ in range(Exp.label_dim[x])
        for z_ in zs_unique
    }

    # P(Y|X,Z)
    p_y_xz = {
        (x_, y_, z_): np.mean(df
                              .loc[lambda df_: df_[x] == x_]
                              .loc[lambda df_: (df_[zs].values == z_).squeeze()]
                              [y]
                              .values == y_)
        for x_ in range(Exp.label_dim[x])
        for y_ in y_unique
        for z_ in zs_unique
    }

    # ATE
    E_y_do_x_0 = 0.0
    E_y_do_x_1 = 0.0

    # E_y_do_x=[]
    # for idx in range(Exp.label_dim[x]):
    #     E_y_do_x.append(0)
    #
    #
    # for y_ in y_unique:
    #     for zs_ in zs_unique:
    #         for x_ in range(Exp.label_dim[x]):
    #             for idx in range(Exp.label_dim[x]):
    #                 E_y_do_x[idx]+= y_ * p_y_xz[(x_, y_, zs_)] * p_z_x[(idx, zs_)] * p_x[x_]

    E_y_do_x = {}

    for idx in range(Exp.label_dim[x]):
        E_y_do_x[idx]={}
        for y_ in y_unique:
            E_y_do_x[idx][y_]=0
            for zs_ in zs_unique:
                for x_ in range(Exp.label_dim[x]):
                    E_y_do_x[idx][y_] += p_y_xz[(x_, y_, zs_)] * p_z_x[(idx, zs_)] * p_x[x_]

    # return E_y_do_x_1 , E_y_do_x_0
    return E_y_do_x



def estiamte_ate_backdoor_direct(Exp, df, x, y, zs):
    """
    Estiamte the ATE of a system from a dataframe of samples `ds`
    using frontdoor adjustment directly on ML estimates of probability.

    E[Y|do(X) = x'] = \sum_{x,y,z} y P[y|x,z] P(z|x') P(X)
    E[Y|do(X) = x'] = \sum_{y,z} P[y|x,z] P(z)

    Arguments
    ---------
    df: pandas.DataFrame
    x: str
    y: str
    zs: list[str]

    Returns
    -------
    ATE: float
    """
    zs_unique = [tuple(a) for a in np.unique(df[zs].values, axis=0)]
    y_unique = np.unique(df[y].values, axis=0)

    # P(Z)
    p_z = {
        z_: np.mean(df[zs[0]] == z_)
        for z_ in range(Exp.label_dim[zs[0]])
    }

    # P(Y|X,Z)
    p_y_xz = {
        (x_, y_, z_[0]): np.mean(df
                              .loc[lambda df_: df_[x] == x_]
                              .loc[lambda df_: (df_[zs].values == z_[0]).squeeze()]
                              [y]
                              .values == y_)
        for x_ in range(Exp.label_dim[x])
        for y_ in y_unique
        for z_ in zs_unique
    }

    # ATE

    E_y_do_x={}
    # for idx in range(Exp.label_dim[x]):
    #     E_y_do_x.append(0)

    for x_ in range(Exp.label_dim[x]):
        E_y_do_x[x_]={}
        for y_ in y_unique:
            E_y_do_x[x_][y_] = 0
            for z_ in zs_unique:
                # for idx in range(Exp.label_dim[x]):
                    # E_y_do_x[idx]+= y_ * p_y_xz[(x_, y_, zs_)] * p_z_x[(idx, zs_)] * p_x[x_]

                    E_y_do_x[x_][y_]+= p_y_xz[(x_, y_, z_[0])] * p_z[z_[0]]

    # return E_y_do_x_1 , E_y_do_x_0
    return E_y_do_x



# x = torch.randint(2, (40000, 3))
# px = pd.DataFrame(x.numpy())
#
# class temp:
#     label_dim={'X':2, 'Z':2, 'Y':2}
#
# px = px.rename(columns={0: 'X', 1: 'Z', 2:'Y'})
# ret=  estiamte_ate_backdoor_direct(temp, px, 'X', 'Y', ['Z'])
#
# # px.rename(columns={'0': 'X', 'Z': 'Y'}, inplace=True)
# print(ret)