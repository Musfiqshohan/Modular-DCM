import itertools
import json
import os

import numpy as np
import torch
# import seaborn as sns
# import matplotlib.pyplot as plt
# import collections
# old constants
from datetime import datetime



class Experiment:

    def __init__(self, exp_name, set_truedag, **kwargs):

        self.exp_name = exp_name
        # self.exp_name = kwargs.get('exp_name', 'conditional-DCGAN')

        self.PLOTS_PER_EPOCH = 1

        ########## For celeba dataset #######
        self.boundaries= None
        self.PROJECT_NAME = kwargs.get('PROJECT_NAME', 'conditional-DCGAN')

        self.NOISE_DIM = kwargs.get('NOISE_DIM', 128)
        self.CONF_NOISE_DIM = kwargs.get('CONF_NOISE_DIM', 128)
        self.generator_decay=1e-6
        self.discriminator_decay=1e-6
        self.IMAGE_NOISE_DIM = kwargs.get('IMAGE_NOISE_DIM', 100)
        self.IMAGE_FILTERS = kwargs.get('IMAGE_FILTERS', [128, 64, 32])
        self.IMAGE_SIZE =  kwargs.get('IMAGE_SIZE', 32)
        self.ENCODED_DIM =  kwargs.get('ENCODED_DIM', 10)

        self.obs_state = kwargs.get('obs_state', 2)

        self.G_hid_dims = kwargs.get('G_hid_dims')  # in_d1  dn_out
        self.D_hid_dims = kwargs.get('D_hid_dims')  # 3x10x5x1
        # G_hid_dims=[10, 25, 25, 10],
        # D_hid_dims=[10, 15, 10, 5],

        # for ett non id
        # G_hid_dims=[30,40,30,20,10],
        # D_hid_dims= [20, 30, 20, 10, 5]

        # G_hid_dims=[30,60,90,60,30,15],
        # D_hid_dims=[20,30,60,30,20,10],

        self.CRITIC_ITERATIONS = kwargs.get('CRITIC_ITERATIONS', 5)
        self.LAMBDA_GP = kwargs.get('LAMBDA_GP', 0.1)  # It was 0.3

        self.learning_rate = kwargs.get('learning_rate', 2 * 1e-5)
        self.betas = (0.5, 0.9)
        self.Synthetic_Sample_Size = kwargs.get('Synthetic_Sample_Size', 20000)
        self.intv_Sample_Size = kwargs.get('intv_Sample_Size', 20000)
        self.ex_row_size = kwargs.get('ex_row_size', 20)
        self.batch_size = kwargs.get('batch_size', 100)  # from 256
        self.intv_batch_size = kwargs.get('intv_batch_size', 100)  # from 256
        self.num_epochs =  kwargs.get('num_epochs', 300)
        self.STOPAGE1 = 50
        self.STOPAGE2 = 20000
        self.lr_dec = 1

        self.curr_epoochs = 0
        self.curr_iter = 0

        # gumbel-softmax
        self.temp_min = kwargs.get('temp_min', 0.00001)
        self.ANNEAL_RATE = 0.000003
        self.start_temp = kwargs.get('Temperature', 0.5)
        self.Temperature = kwargs.get('Temperature', 0.5)

        self.dataset_activated = kwargs.get('dataset_activated', [0])

        # Data_intervs=[{}, {"X1":1,"W":1}, {"X1":1,"W":0}, {"X1":0,"X2":0}]

        self.SAVE_MODEL = True
        self.LOAD_MODEL = False
        self.LOAD_TRAINED_CONTROLLER = False
        self.load_which_models={}
        self.pre_trained_by_others = []
        self.checkpoints = {}

        # self.DEVICE = get_freer_gpu()
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        now = datetime.now()
        self.curDATE = now.strftime("%b_%d_%Y")
        self.curTIME = now.strftime("%H_%M")


        dlist=[]
        self.Data_intervs = kwargs.get('Data_intervs', dlist)
        self.Data_observs = kwargs.get('Data_observs', [])
        self.num_datasets = len(self.Data_intervs)


        self.G_avg_losses = []
        self.D_avg_losses = []

        # scm ground truth
        self.noise_states = kwargs.get('noise_states', 8)
        self.latent_state = kwargs.get('latent_state', 8)
        self.dist_thresh = kwargs.get('dist_thresh', 0.2)
        self.allowed_noise = kwargs.get('allowed_noise', 0.50)

        self.causal_hierarchy = kwargs.get('causal_hierarchy', 1)

        # self.evaluate_after_epochs = kwargs.get('sachsEvaluation', None)


        ret = set_truedag(self.noise_states, self.latent_state, self.obs_state, self.Data_intervs)
        self.DAG_desc, self.Complete_DAG_desc, self.Complete_DAG, self.complete_labels, self.Observed_DAG, self.label_names, self.image_labels, self.rep_labels, self.interv_queries, self.cf_queries, self.latent_conf, \
        self.confTochild, self.exogenous, self.cf_intervene, self.cf_observe, self.cf_evidence, self.cflabel_names, self.twin_map, self.Twin_Network, self.cf_exogenous, \
        self.noise_params, self.train_mech_dict, self.label_dim, self.plot_title \
            = ret


        self.true_bn = kwargs.get('true_bn', None)
        self.features= kwargs.get('features', ["digit", "thickness", "color"])



        self.cf_samples = self.Synthetic_Sample_Size
        self.num_labels = len(self.label_names)

        main_path= kwargs.get('main_path', f"./SAVED_EXPERIMENTS")

        # saving model and results

        self.new_experiment= kwargs.get('new_experiment', True)


        if self.new_experiment == True:
            os.makedirs(main_path ,exist_ok=True)
            saved_path = main_path + "/" + self.exp_name
            self.SAVED_PATH = kwargs.get('SAVED_PATH', saved_path)

            self.LOAD_MODEL_PATH = kwargs.get('LOAD_MODEL_PATH', self.SAVED_PATH)

            INSTANCES = {}
            INSTANCES["last_exp"] = self.SAVED_PATH
            with open(main_path +"/SHARED_INFO.txt", 'w') as fp:
                fp.write(json.dumps(INSTANCES))


        self.file_roots = main_path + self.Complete_DAG_desc + "/preprocessed_dataset/"


        self.isJoint = False
        self._data_sampler = None
        self.test_marginals=False
        self.bayesNet= None


    def anneal_temperature(self, tot_iters):

        # if (tot_iters) % 100 == 1:
        self.Temperature = np.maximum(
            self.Temperature * np.exp(-self.ANNEAL_RATE * tot_iters),
            self.temp_min)
        print(tot_iters, ":Temperature", self.Temperature)
