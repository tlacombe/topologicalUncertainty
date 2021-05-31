import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.linalg import eigh
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph
import itertools

from scipy.sparse import csgraph

import os

#####################
### Graph as data ###
#####################
def hks_signature(eigenvectors, eigenvals, time):
    return np.square(eigenvectors).dot(np.diag(np.exp(-time * eigenvals))).sum(axis=1)


def build_csv(dataset_name, pad_size=30):
    list_hks_times = [10.0]
    path_dataset = './%s/' %dataset_name

    features = pd.DataFrame(index=range(len(os.listdir(path_dataset + "mat/"))), 
                            columns=["label"] + ["eval" + str(i) for i in range(pad_size)] 
                            + [name + "-percent" + str(i) for name, i in itertools.product([str(f) for f in list_hks_times], 10 * np.arange(11))])

    for idx, graph_name in enumerate((os.listdir(path_dataset + "mat/"))):
        name = graph_name.split("_")
        gid = int(name[name.index("gid") + 1]) - 1
        A = np.array(loadmat(path_dataset + "mat/" + graph_name)["A"], dtype=np.float32)
        num_vertices = A.shape[0]
        label = int(name[name.index("lb") + 1])
        L = csgraph.laplacian(A, normed=True)
        egvals, egvectors = eigh(L)

        eigenvectors = np.zeros([num_vertices, pad_size])
        eigenvals = np.zeros(pad_size)
        eigenvals[:min(pad_size, num_vertices)] = np.flipud(egvals)[:min(pad_size, num_vertices)]
        eigenvectors[:, :min(pad_size, num_vertices)] = np.fliplr(egvectors)[:, :min(pad_size, num_vertices)]
        graph_features = []
        graph_features.append(eigenvals)

        for hks_time in list_hks_times:
            # features
            graph_features.append(np.percentile(hks_signature(eigenvectors, eigenvals, time=hks_time), 
                                                10 * np.arange(11)))
        features.loc[gid] = np.insert(np.concatenate(graph_features), 0, label)
    features['label'] = features['label'].astype(int)

    features.to_csv(path_dataset + dataset_name + ".csv")


def save_adjacency(A, gid, label, path):
    mat_name = "nodes_%i_edges_%i_gid_%i_lb_%i_index_1_adj.mat" % (A.shape[0], int(np.sum(A > 0)), gid, label)
    mat_file = {
        '__header__': 'PYTHON mimick MAT-file format',
        '__version__': 'nc',
        '__globals__': [],
        'A': A.astype(int)
    }
    return savemat(file_name=path + mat_name, mdict=mat_file)


def get_params(dataset):
    path_dataset = 'COX2/'
    mat = [np.array(loadmat(dataset + "/mat/" + graph_name)["A"], dtype=np.float32) for graph_name in os.listdir(dataset + "/mat/")]
    nums_vertices = [A.shape[0] for A in mat]
    nums_edges = [np.sum(A) / 2 for A in mat]
    return nums_vertices, nums_edges


def generate_fake_graphs(dataset, n_graphs=100):
    path = 'Fake-' + dataset + '/mat/'
    nodes_distrib, edges_distrib = get_params(dataset)
    nb_graphs = len(nodes_distrib)
    label = 0   # Do not play a role here
    for gid in range(n_graphs):
        n = nodes_distrib[np.random.randint(nb_graphs)]
        m = edges_distrib[np.random.randint(nb_graphs)]
        p = m / (n * n)  # proportion of edges for ER-graphs
        G = erdos_renyi_graph(int(n), p)
        A = nx.adjacency_matrix(G).todense()
        save_adjacency(A, gid+1, label, path)



