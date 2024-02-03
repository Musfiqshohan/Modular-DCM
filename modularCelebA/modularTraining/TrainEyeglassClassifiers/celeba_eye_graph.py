# from ModularUtils.FunctionsConstant import getdoKey
# from ModularUtils.ControllerConstants import generate_permutations

import modularTraining.constant_paths as const

class CausalGraph():

    def __init__(self, name, dag, confs):
        self.DAG_desc = name

        self.Complete_DAG_desc = name
        self.Observed_DAG = dag

        self.num_confs = len(confs.keys())
        self.Complete_DAG = {}
        for cnf in range(self.num_confs):
            self.Complete_DAG["U" + str(cnf)] = []

        self.latent_conf = {}
        for var in self.Observed_DAG:
            self.Complete_DAG[var] = []
            self.latent_conf[var] = []

        self.confTochild = confs

        for cnf in self.confTochild:
            for var in self.confTochild[cnf]:
                self.latent_conf[var].append(cnf)
                self.Complete_DAG[var].append(cnf)

        for var in self.Observed_DAG:
            self.Complete_DAG[var] = self.Complete_DAG[var] + self.Observed_DAG[var]

        self.complete_labels = list(self.Complete_DAG.keys())
        self.label_names = list(self.Observed_DAG.keys())



        self.image_labels= None
        self.rep_labels= None




def set_celeba_eye(noise_states, latent_state, obs_state, Data_intervs):

    Observed_DAG = {
        "Male": [],
        "Eyeglasses": [],
    }

    confTochild = {"U0": ["Male", "Eyeglasses"]}
    G = CausalGraph(name="celeba", dag=Observed_DAG, confs=confTochild)

    plot_title = "celeba image experiment"
    #
    G.image_labels = []
    G.rep_labels = []

    label_dim={}
    for lb in G.label_names:
        label_dim[lb]=2

    interv_queries = []

    exogenous = {}
    for label in G.label_names:
        if label not in G.image_labels:
            exogenous[label] = "n" + label


    train_mech_dict={}

    return G.DAG_desc, G.Complete_DAG_desc, G.Complete_DAG, G.complete_labels, G.Observed_DAG, G.label_names, label_dim, G.image_labels, G.rep_labels, interv_queries, G.latent_conf, \
           G.confTochild, exogenous, train_mech_dict,  plot_title





