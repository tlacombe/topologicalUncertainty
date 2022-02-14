# This file is released under MIT licence, see file LICENSE.
# Author(s):       Theo Lacombe
#
# Copyright (C) 2021 Inria

import numpy as np
import gudhi as gd
try:
    import tensorflow as tf
except ImportError:
    print("Cannot import tensorflow. build_graphs_from_deep_model will not be available.")


def _matrix_and_input(x_train, W):
    '''
    :param x_train: a N x D dataset (N points, dimension D). Can be the "latent dataset" in a hidden layer.
    :param W: a D x D' weight matrix.
    :return: activated weights, i.e. (x_train[k][i] W[i,j]) for i=1..D, for k = 1..N
    '''
    return np.array([np.dot(np.diag(x), W) for x in x_train])


def _get_vertices_and_edges_values_from_input(model, x_train, layers_id=None, only_fc=True):
    '''
    :param model: a tensorflow model
    :param x_train: n_train x d0 input dataset.
    :param layers_id: a iterable (list, set...) of ids for the layers we want to focus on. If `None` (default) all
                      (fc) layers are taken into account.
    :param only_fc: Only consider fully connected layers ("dense" in tensorflow). Only `True` available as of now.
    :return: a list of edges values. `X[ell][j]` gives the values of the `layers_id[ell]`-th layer
             for the j-th data of (dim d_ell).
             and a list of weight values. W[ell][j] gives the values of the ell-to-(ell+1)-th matrix for the j-th data.
    '''
    nodes_values = []
    edges_values = []
    current_x = x_train

    if layers_id is None:
        layers_id = np.arange(len(model.layers))

    if only_fc:
        for id, layer in enumerate(model.layers):
            if ('dense' in layer.name) and (id in layers_id):
                nodes_values.append(current_x)
                W = layer.weights[0].numpy()
                weighted_edges = _matrix_and_input(current_x, W)
                edges_values.append(weighted_edges)
            current_x = tf.keras.models.Sequential([layer]).predict(current_x)
        nodes_values.append(current_x)
    else:
        raise NotImplemented('Only fully-connected layers are available in this version yet. Set `only_fc=True`.')

    return edges_values  # nodes_values,


def _simplex_tree_from_bipartite_matrix(W):
    '''
    Turn a (numpy) matrix representing a bipartite graph into a SimplexTree (gudhi object)
    from which we can extract topological information.

    :param W: A `numpy` matrix representing a bipartite graph (W.shape[0] vertices connected to W.shape[1] vertices).
    :return: A `SimplexTree` G representing the corresponding weighted (bipartite) graph.
    '''
    G = gd.SimplexTree()

    # We make sure the vertices are there "from the start" in the filtration. 
    for i in np.arange(0, W.shape[0] + W.shape[1]):
        G.insert([i], filtration=-np.inf)

    for i in np.arange(0, W.shape[0]):
        for j in np.arange(W.shape[0], W.shape[0] + W.shape[1]):
            G.insert([i, j], filtration= -np.abs(W[i, j - W.shape[0]]))

    return G


def build_graphs_from_adjacency_matrices(matrices):
    raise NotImplemented('Diagrams from adjacency matrices will be provided in a future version.')


def build_graphs_from_deep_model(model, x_train, layers_id=None):
    '''
    Given a tensorflow sequential model, a set of observations and a subset of layers to consider,
    compute a list of list of graphs (encoded as SimplexTree) for the different layers and observations.

    :param model: A tensorflow (sequential) model.
    :param x_train: The set of observations used to compute Fréchet means.
    :param layers_id: a iterable (list, set...) of ids for the layers we want to focus on. If `None` (default) all
                      (fc) layers are taken into account.
    :return: a list of list of graphs. `graphs[ell][id_train]` represents the activation graph of the `id_train`-th
             observation at the  `layers_id[ell]`-th layer.
    '''

    graphs = []
    # lists_of_adjacency_matrices = []

    weights_tmp = _get_vertices_and_edges_values_from_input(model, x_train, layers_id)

    weights = [[W[k] for W in weights_tmp] for k in range(len(x_train))]

    for weights_per_layer in weights:
        graphs_per_layers = []

        for W in weights_per_layer:
            G = _simplex_tree_from_bipartite_matrix(W)
            graphs_per_layers.append(G)

        graphs.append(graphs_per_layers)

    return graphs
