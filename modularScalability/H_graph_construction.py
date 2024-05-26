import copy

import matplotlib.pyplot as plt
from itertools import chain

from ConstantFunctions import draw_true_graph, build_compares, getdoKey, asKey
from ControllerConstants import generate_permutations


def get_ccom(latent_conf, confTochild, sub_graph, visited, cur_node):

    nbrs=[]
    for conf in latent_conf[cur_node]:
        nbrs+= confTochild[conf]

    for nbr in nbrs:
        if (nbr in sub_graph) and (nbr not in visited):
            visited.append(nbr)
            get_ccom(latent_conf, confTochild, sub_graph, visited, nbr)

def get_anc(Observed_DAG,sub_graph, visited, cur_node):

    for par in Observed_DAG[cur_node]:
        if (par in sub_graph) and (par not in visited):
            # print(f'node:{cur_node} <- par:{par}')
            visited.append(par)
            get_anc(Observed_DAG, sub_graph , visited, par)

def get_des(Observed_DAG,sub_graph, visited, cur_node):

    for ch in Observed_DAG:
        if (cur_node in Observed_DAG[ch]) and (ch not in visited):
            visited.append(ch)
            get_des(Observed_DAG, sub_graph , visited, ch)


def Find_MACS(Observed_DAG, latent_conf, confTochild, prev_allowed, Y):
    # while(set(prev_allowed) != set(allowed_sub_graph)):

    c_components = [Y]
    get_ccom(latent_conf, confTochild, Observed_DAG.keys(), c_components, Y)
    c_comps = sorted(c_components)

    allowed_sub_graph=[]
    for cc in c_comps:
        visited=[cc]
        get_des(Observed_DAG, Observed_DAG.keys(), visited, cc)
        allowed_sub_graph+=visited


    ancestors=[Y]
    get_anc(Observed_DAG, allowed_sub_graph, ancestors, Y)
    allowed_sub_graph = sorted(ancestors)


    # print("Output MACS", allowed_sub_graph)
    return  allowed_sub_graph

def hasBackdoor(Observed_DAG, mediator, node2, visited):
    visited.append(mediator)
    if mediator in node2:
        return True

    ret= False
    for par in Observed_DAG[mediator]:
        if par not in visited:
            ret= ret or hasBackdoor(Observed_DAG, par, node2, visited)

    return ret




def merge_nodes(H_graph, ndlist):
    # print(set(H_graph[nd1]),set(H_graph[nd2]))
    set_list=[]
    for ndi in ndlist:
        set_list.append(set(tuple(row) for row in H_graph[ndi]))

    uset = list(set().union(*set_list))

    for ndi in ndlist:
        uset.remove(tuple(ndi))

    new_node = tuple()
    for ndi in ndlist:
        new_node+=ndi

    H_graph[new_node] = [list(u) for u in uset]

    for ndi in ndlist:
        del H_graph[ndi]  # removing old node from H graph

    new_ndlist=[]
    for ndi in ndlist:
        new_ndlist.append(list(ndi))


    for nd in H_graph:  # removing old hnodes as parents and adding new hnode as parent
        for ndi in new_ndlist:
            if ndi in H_graph[nd]:
                H_graph[nd].remove(ndi)
                if new_node not in H_graph[nd]:
                    H_graph[nd].append(list(new_node))

    return H_graph,new_node


def check_cycle_lenth2(H_graph):

    hnodes= list(H_graph.keys())
    i=0
    while i < len(hnodes):
        j=i+1
        while j < len(hnodes):
            nd1,nd2=hnodes[i],hnodes[j]
            if set(nd1)== set(nd2) or nd1 not in H_graph or nd2 not in H_graph:
                #if they are same nodes then skipping. Or if they are nodes which have been already merged
                pass

            elif list(nd1) in H_graph[tuple(nd2)] and list(nd2) in H_graph[tuple(nd1)]:
                H_graph, new_node= merge_nodes(H_graph, [nd1, nd2])
                hnodes.append(new_node)

            j+=1
        i+=1

    return H_graph


def check_cycle_lenth3(H_graph):

    hnodes= list(H_graph.keys())
    # for i in range (len(hnodes)):
    i = 0
    while i < len(hnodes):
        # for j in range(i+1, len(hnodes)):
        j = i + 1
        while j < len(hnodes):
            k = j + 1
            while k < len(hnodes):
        #     for k in range(j + 1, len(hnodes)):
                nd1,nd2, nd3 =hnodes[i],hnodes[j], hnodes[k]
                if set(nd1)== set(nd2) or set(nd2)== set(nd3) or set(nd1)== set(nd3) or nd1 not in H_graph or nd2 not in H_graph or nd3 not in H_graph:
                #if they are same nodes then skipping. Or if they are nodes which have been already merged
                    pass

                elif (list(nd1) in H_graph[tuple(nd2)] and list(nd2) in H_graph[tuple(nd3)] and list(nd3) in H_graph[tuple(nd1)]) or\
                        (list(nd3) in H_graph[tuple(nd2)] and list(nd2) in H_graph[tuple(nd1)] and list(nd1) in H_graph[tuple(nd3)]):
                    H_graph,new_node= merge_nodes(H_graph, [nd1, nd2, nd3])

                k+=1
            j+=1
        i+=1

    return H_graph




def get_H_graphs(intervs, latent_conf, confTochild, Complete_DAG, Observed_DAG, label_names):
    latent_conf=copy.deepcopy(latent_conf)
    confTochild=copy.deepcopy(confTochild)
    Observed_DAG=copy.deepcopy(Observed_DAG)
    Complete_DAG=copy.deepcopy(Complete_DAG)

    for intv in intervs:
        Observed_DAG[intv]=[]
        Complete_DAG[intv]=[]
        latent_conf[intv]=[]

    for conf in confTochild:
        confTochild[conf]= list(set(confTochild[conf]) - set(intervs))

    h_nodes=[]
    for label in label_names:
        if label not in sum(h_nodes, []):
            visited = [label]
            get_ccom(latent_conf, confTochild, Observed_DAG, visited, label)
            h_nodes.append(copy.deepcopy(visited))
            # print(visited)


    H_graph = {}
    for node1 in h_nodes:  #for having node1-> node2
        for node2 in h_nodes:
            if tuple(node2) not in H_graph:
                H_graph[tuple(node2)] = []

            if set(node1) == set(node2) :
                continue

            parents = []  # find parents of node2
            for lb in node2:
                # ret=[par for par in Observed_DAG[lb] if lb not in node2]
                ret = list(set(Observed_DAG[lb]) - set(node2) - set(parents))
                # print(ret)
                parents += ret
            # print('node2',node2, 'parents', parents)
            # parents= list(chain(*[Observed_DAG[lb] for lb in node2]))

            test=0
            for v in set(node1) & set(parents):  # iterate over those parents which belong to node1
                if hasBackdoor(Observed_DAG, v, node2, []) == True:
                    H_graph[tuple(node2)].append(node1)


    H_graph= check_cycle_lenth2(H_graph)
    H_graph= check_cycle_lenth3(H_graph)

    return H_graph


def get_parents(Observed_DAG, cur_joint ):
    parents = []
    for nd in cur_joint:
        parents+=Observed_DAG[nd]
    return list(set(parents) - set(cur_joint))

def Find_Joint(latent_conf, confTochild, Observed_DAG, H_graph, cur_joint):

    prev_joint= []

    prev_joint= copy.deepcopy(cur_joint)

    parents = get_parents(Observed_DAG, cur_joint)
    _A=[]
    for par in parents:
        if hasBackdoor(Observed_DAG,  par, cur_joint, [])==True:
            _A.append(par)
    cur_joint= list(set(cur_joint + _A))

    T_y=[]
    for var in cur_joint:
        T_y += Find_MACS(Observed_DAG,latent_conf, confTochild, [],  var)

    cur_joint = list(set(cur_joint + T_y))


    # if len(_A)==0 and len(set(T_y))==len(set(cur_joint)):
    #     return cur_joint
    # cur_joint= list(set(cur_joint + _A).union(set(T_y)))

    if set(prev_joint) == set(cur_joint):
        return sorted(cur_joint)

    return Find_Joint(latent_conf, confTochild, Observed_DAG, H_graph, cur_joint)






def set_LargeGraph(noise_states, latent_state, obs_state, Data_intervs):
    DAG_desc = "LargeGraph"

    Complete_DAG_desc = "LargeGraph"
    Complete_DAG = {}
    plot_title="Modular Training distribution convergence"

    Observed_DAG = {
        '1':[], '2':['1'], '3':['2'],
        '4':[],
        '5': ['2'], '6': ['7','8'],  '7': ['4'], '8': ['5', '11'],'9': ['6', '19'], '10': ['16'],'11': ['10'],
        '12':[],
        '13':[],
        '14':['8','15'], '15':['12','27'], '16':[],
        '17':['9','18','27'], '18':['19','22'], '19':['13'], '20':['17','21'], '21':['22'], '22':['19'],
        '23':['14','24','27','28'], '24':[], '25':['17','27'], '26':['23','27','29'], '27':['24'],
        '28':['29'],
        '29':[]
    }

    confTochild = {"U0": ["1", "3"], "U1": ["5", "6"], "U2": ["7", "9"], "U3": ["8", "10"], "U4": ["14", "15"],
                   "U5": ["15", "16"], "U6": ["17", "18"], "U7": ["18", "19"], "U8": ["20", "21"], "U9": ["21", "22"],
                   "U10": ["23", "24"], "U11": ["24", "25"], "U12": ["26", "27"],
                   "U13":["9","11"]}

    label_names = list(Observed_DAG.keys())

    num_confounders= len(confTochild.keys())
    Complete_DAG = {}
    for conf in range(num_confounders):
        Complete_DAG["U"+str(conf)] = []

    latent_conf={}
    for var in Observed_DAG:
        Complete_DAG[var]=[]
        latent_conf[var] = []



    for conf in confTochild:
        for var in confTochild[conf]:
            latent_conf[var].append(conf)
            Complete_DAG[var].append(conf)

    # visited=[]
    # while len(visited)!=len(label_names):
    for var in Observed_DAG:
        Complete_DAG[var]=Complete_DAG[var]+ Observed_DAG[var]

    complete_labels = list(Complete_DAG.keys())


    image_labels= []
    rep_labels=[]

    label_dim = {}
    for label in Observed_DAG.keys():
        label_dim[label] =  obs_state


    for conf in confTochild:
        label_dim[conf] = latent_state

    interv_queries = []

    exogenous = {}
    for label in label_names:
        if label not in image_labels:
            exogenous[label] = "n" + label



    noise_params = {}
    for label in Observed_DAG:
        noise_params["n" + label] = (0.5, noise_states)

    for conf in confTochild:
        noise_params[conf] = (0.1, latent_state)


    train_mech_dict={}

    H_graph={}
    # for intv in Data_intervs:
    for intv in Data_intervs:
        print(f'Intv:{intv}')
        H_graph[asKey(intv)]= get_H_graphs(intv.keys(), latent_conf, confTochild, Complete_DAG, Observed_DAG, label_names)
        print(f'Hgraph: {H_graph[asKey(intv)]}')



    S=[]
    O=[]
    for hnode in H_graph[asKey({})]:
        print('At hnode:', hnode)

        for intv in Data_intervs:


            ulatent_conf = copy.deepcopy(latent_conf)
            uconfTochild = copy.deepcopy(confTochild)
            uObserved_DAG = copy.deepcopy(Observed_DAG)

            for int in intv:
                uObserved_DAG[int] = []
                ulatent_conf[int] = []

            for conf in uconfTochild:
                uconfTochild[conf] = list(set(confTochild[conf]) - set(intv))

            A= Find_Joint(ulatent_conf, uconfTochild, uObserved_DAG, H_graph[asKey(intv)], list(hnode))
            do_par= get_parents(uObserved_DAG, A)

            if set(intv.keys()) & set(hnode)== set({}):
                print('Found Joint to train ++++observation: ', A, 'do_pars',do_par, 'intv:',intv)
                O.append((A,do_par, intv))  #do a minimum check now or later

            else:
                print('Found Joint to train ////intervention: ', A, 'do_pars', do_par, 'intv:', intv)
                S.append((A,do_par, intv))  #do a minimum check now or later



    for label in Observed_DAG:
        if label not in image_labels:
            label_dim["n" + label] = noise_states

    return DAG_desc, Complete_DAG_desc, Complete_DAG, complete_labels, Observed_DAG, label_names, image_labels, rep_labels, interv_queries,  latent_conf, \
           confTochild, exogenous, noise_params, train_mech_dict, label_dim, plot_title




if __name__ == '__main__':

    #Returns an H-graph for a specific large causal graph.
    # G= set_LargeGraph(2, 2, 2, [{},{'17':0}])  # when you have intervention on node 17.
    G= set_LargeGraph(2, 2, 2, [{}]) # no intervention.
