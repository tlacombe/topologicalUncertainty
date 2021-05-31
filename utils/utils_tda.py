# This file is released under MIT licence, see file LICENSE.
# Author(s):       Theo Lacombe
#
# Copyright (C) 2021 Inria

import numpy as np
import utils.utils_graphs as ug


def dgm_per_layers_from_graphs(graphs_per_layers):
    # Assume backend is gudhi by default.
    # Note: we add negative wheights and then flip the diagram to implicitely perform superlevel sets persistence.
    # Named after B.Rieck
    [G.compute_persistence(min_persistence=-1.) for G in graphs_per_layers]
    dgms = [G.persistence_intervals_in_dimension(0) for G in graphs_per_layers]
    dgms = [- dgm[np.where(np.isfinite(dgm[:, 1]))] for dgm in dgms]
    return dgms


def diags_from_graphs(graphs):
    return [dgm_per_layers_from_graphs(gpl) for gpl in graphs]


def wasserstein_barycenter_1D(point_clouds):
    '''
    Assume N array with the same number of points K, returns the naive 1D barycenter with support of size K
    '''
    s = np.sort(point_clouds)
    return np.mean(s, axis=0)


def wasserstein_distance_1D(a, b, p=2., average=True):
    n = a.shape[0]
    if np.isinf(p):
        res = np.max(np.abs(np.sort(a) - np.sort(b)))
    else:
        res = np.sum(np.abs(np.sort(a) - np.sort(b))**p)**(1./p)
    if average:
        return res / n
    else:
        return res


def topological_uncertainty(model, x, all_barycenters, layers_id=None, aggregation='mean', p=2., average_wasserstein=True):
    nlayer = len(all_barycenters[0])  # number of layers used
    predicted_classes = np.argmax(model.predict(x), axis=-1)
    graphs = ug.build_graphs_from_deep_model(model, x, layers_id=layers_id)
    diags = diags_from_graphs(graphs)

    res = np.array([[wasserstein_distance_1D(diags_per_layer[ell][:,1], all_barycenters[predicted_class][ell], p=p, average=average_wasserstein) for ell in range(nlayer)]
                     for (diags_per_layer, predicted_class) in zip(diags, predicted_classes)])

    if aggregation=='mean':
        return np.mean(res,axis=1)
    elif aggregation=='max':
        return np.max(res,axis=1)
    elif aggregation is None:
        return res
    else:
        raise ValueError('aggregation=%s is not valid. Set it to mean (default) or max.')


def bary_of_set_from_deep_model(model, x, layers_id=None):
    graphs = ug.build_graphs_from_deep_model(model, x, layers_id=layers_id)
    diags = diags_from_graphs(graphs)
    nlayer = len(diags[0])  # number of layer (with weight matrix) that we use from our model.
    point_clouds_per_layer = [np.array([diag[ell][:,1] for diag in diags]) for ell in range(nlayer)]

    wbarys = [wasserstein_barycenter_1D(pc) for pc in point_clouds_per_layer]

    return wbarys