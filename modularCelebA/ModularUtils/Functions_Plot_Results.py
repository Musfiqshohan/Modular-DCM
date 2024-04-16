import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from ModularUtils.FunctionsConstant import getdoKey, plot_lines




def concat_vertical(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    # max_height = np.max([ha, hb])
    # total_width = wa+wb
    max_width = np.max([wa, wb])
    total_height = ha+hb
    new_img = np.zeros(shape=(total_height,max_width, 3))
    new_img[:ha,:wa]=imga
    new_img[ha:ha+hb,:wb]=imgb
    # new_img[:hb,wa:wa+wb]=imgb
    return new_img


def concat_horizon(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img






def plot_saved_results(Exp, last_exp, epochs, delta, pre_labels=None, benchmarks=None):


    if last_exp== None:
        SHARED_INFO = "/local/scratch/a/rahman89/PycharmProjects/conditional-DCGAN/SAVED_EXPERIMENTS/" + Exp.Complete_DAG_desc + "/SHARED_INFO.txt"
        with open(SHARED_INFO) as f:
            data = f.read()
        INSTANCE = json.loads(data)

        last_exp = INSTANCE["last_exp"]
    print(last_exp)

    tvd_diff = {}
    kl_diff = {}



    for query_list in Exp.interv_queries:
        for intv in query_list["intervs"]:
            query = getdoKey(query_list["obs"], intv)

            # if query=='P(medC|do(medD_1))':
            #     print('query', query)
            #     continue

            tvd_diff[query] = []
            kl_diff[query] = []

            if benchmarks!=None:
                for bnch in benchmarks:
                    tvd_diff[bnch[0]+query] = []
                    kl_diff[bnch[0]+query] = []

        # tvd_diff[query_list['expr']] = []
        # kl_diff[query_list['expr']] = []



    # for query in Exp.cf_queries:
    #     tvd_diff[query["expr"]] = []
    #     kl_diff[query["expr"]] = []

    print("tvd diffs")
    for dist in tvd_diff:
        if dist[0]=="P":
            try:
                tvd_diff[dist] = torch.load(last_exp + "/tvd/" + dist).detach().cpu().numpy()
                kl_diff[dist] = torch.load(last_exp + "/kl/" + dist).detach().cpu().numpy()
            except FileNotFoundError:
                print("Wrong file or file path")
                return



        # bench_exp = "/local/scratch/a/rahman89/PycharmProjects/conditional-DCGAN/SAVED_EXPERIMENTS/"+ Exp.Complete_DAG_desc +"/benchmark_result"
        if benchmarks!= None:
            for bnch in benchmarks:
                bench_exp = bnch[1]
                if Path(bench_exp+ "/tvd/" + dist).is_file() and benchmarks!=None:
                        tvd_diff[bnch[0]+dist] = torch.load(bench_exp + "/tvd/" + dist).detach().cpu().numpy()
                        kl_diff[bnch[0]+dist] = torch.load(bench_exp + "/kl/" + dist).detach().cpu().numpy()

    # print(tvd_diff)
    label_keys = tvd_diff.keys()


    tvd_error, kl_error={}, {}
    new_tvd = {}
    new_kl = {}
    xaxis = []
    for dist in tvd_diff:
        new_tvd[dist], new_kl[dist] = [], []
        tvd_error[dist], kl_error[dist] = [], []
        idx = 0
        while (idx + 1) * delta <= min(epochs, tvd_diff[dist].shape[0]):
            st, en= idx * delta, (idx + 1) * delta
            new_tvd[dist].append(np.mean(tvd_diff[dist][st: en]))
            new_kl[dist].append(np.mean(kl_diff[dist][st: en]))

            # tvd
            error=  abs(tvd_diff[dist][idx * delta: (idx + 1) * delta] - new_tvd[dist][-1])
            tvd_error[dist].append(np.mean(error))

            # kl
            error=  abs(kl_diff[dist][idx * delta: (idx + 1) * delta] - new_kl[dist][-1])
            kl_error[dist].append(np.mean(error))

            idx += 1

        xaxis = [i * delta for i in range(len(new_tvd[dist]))]

    label_keys = list(tvd_diff.keys())
    if pre_labels!=None:
        label_keys=pre_labels





    bnc1_labels=[]
    bnc2_labels=[]
    if benchmarks!=None and len(benchmarks)==1:
        bnc1_labels= label_keys[1::2]
    elif benchmarks!=None and len(benchmarks)==2:
        bnc1_labels = label_keys[1::3]
        bnc2_labels = label_keys[2::3]


    plot_lines(Exp.plot_title, "Total Variation Distance",
               list(new_tvd.values()), xaxis,
               list(label_keys), bnc1_labels  , bnc2_labels, list(tvd_error.values())  ,save_plot=False,  #odd positions hold the benchmarks
               path=last_exp)

    plot_lines(Exp.plot_title, "KL Divergence",
               list(new_kl.values()), xaxis,
               list(label_keys),  bnc1_labels, bnc2_labels, list(kl_error.values()),  save_plot=False,
               path=last_exp)


# ['$ncmP(V)$','$ncmP(Mek|do[PKA=2])$','$ncmP(Akt|do[PKA=2])$']

