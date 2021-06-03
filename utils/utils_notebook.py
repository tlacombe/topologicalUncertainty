# This file is released under MIT licence, see file LICENSE.
# Author(s):       Theo Lacombe
#
# Copyright (C) 2021 Inria

import matplotlib.pyplot as plt
import numpy as np
import utils.utils_tda as utda

def plot_TU_and_confidence(distribs, confidences, dataset, dataset_ood):
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    L = ['(a)', '(b)'] #, '(c)', '(d)']

    dists_true, dists_fake, dists_2 = distribs
    confs_true, confs_fake, confs_2 = confidences

    axs[0].hist(dists_true, color='blue', alpha=1, label='True-%s (Train)' %dataset, weights = np.ones_like(dists_true)/len(dists_true))
    axs[0].hist(dists_fake, color='green', alpha=0.5, label='Fake-%s (OOD)' %dataset, weights = np.ones_like(dists_fake)/len(dists_fake))
    axs[0].hist(dists_2, color='red', alpha=0.5, label='%s (OOD)' %dataset_ood, weights = np.ones_like(dists_2)/len(dists_2))
    axs[0].set_xlabel('%s Distrib.~of TU (Train: %s)' %(L[0],dataset), fontsize=22)
    axs[0].legend(fontsize=20)

    axs[1].hist(confs_fake, color='green', alpha=1, label='Fake-%s (OOD)' %dataset, bins = np.linspace(0,1,11), weights=np.ones_like(confs_fake)/len(confs_fake))
    axs[1].hist(confs_2, color='red', alpha=1, label='%s (OOD)' %dataset_ood, bins = np.linspace(0,1,11), weights=np.ones_like(confs_2)/len(confs_2))
    axs[1].hist(confs_true, color='blue', alpha=1, label='True-%s (Train)' %dataset, bins = np.linspace(0,1,11), weights=np.ones_like(confs_true)/len(confs_true))
    axs[1].set_xlabel('%s Distrib.~of Confidences (Train: %s)' %(L[1], dataset), fontsize=22)
    axs[1].legend(fontsize=20)

    [ax.grid() for ax in axs]


def plot_TU_circles(distrib_clean, distrib_noisy, sigmas):
    ids = [0, 1, 2, 4] #, 10, 19]
    fig, ax = plt.subplots()
    ax.hist(distrib_clean, color='blue', alpha=1, label='Clean diags', weights = np.ones_like(distrib_clean)/len(distrib_clean))
    for id, dist in zip(ids, distrib_noisy[ids]):
        ax.hist(dist, alpha=0.5, label='$\sigma=%s$' %sigmas[id], weights = np.ones_like(dist)/len(dist))
    ax.legend()
    ax.grid()


def sample_adjancency_matrix(n):
    tmp = np.random.rand(n, n)
    A = (tmp + tmp.T) / 2
    return A


def sample_circle(n, r, noise):
    thetas = 2 * np.pi * np.random.rand(n)
    X = r * np.array([np.cos(thetas), np.sin(thetas)]).T + noise * np.random.randn(n, 2)
    return X




def visu_notebook(circles, diags, barycenter, noisy_circles, diags_of_noisy_circles, sigmas):
    n_row = 5
    xlim = -0.1, 0.5
    fig, axs = plt.subplots(n_row, 5, figsize=(20, 5))
    for i in range(n_row):
        axs[i, 0].scatter(circles[i, :, 0], circles[i, :, 1], color='blue', marker='o')
        axs[i, 0].set_xlim(-1.5, 1.5)
        axs[i, 0].set_ylim(-1.5, 1.5)
        axs[i, 0].set_axis_off()
        axs[i, 0].set_aspect('equal')

        utda.plot_1d_diagram(diags[i], axs[i, 1], xlim=xlim)

        axs[i, 3].scatter(noisy_circles[i, 0, :, 0], noisy_circles[i, 0, :, 1], color='blue', marker='o')
        if i == 0:
            axs[i, 3].set_title('Diagrams of noisy circles \n $\sigma$ = %s' % np.round(sigmas[i], 1))  # , rotation=90
        else:
            axs[i, 3].set_title('$\sigma$ = %s' % np.round(sigmas[i], 1))  # , rotation=90
        axs[i, 3].set_xlim(-1.5, 1.5)
        axs[i, 3].set_ylim(-1.5, 1.5)
        axs[i, 3].set_axis_off()
        axs[i, 3].set_aspect('equal')
        utda.plot_1d_diagram(diags_of_noisy_circles[i][0], axs[i, 4], xlim=xlim)
    for i in np.arange(1, n_row):
        axs[i,2].set_axis_off()
    utda.plot_1d_diagram(barycenter, axs[0, 2], color='red', xlim=(-0.1, 0.5))

    axs[0, 0].set_title('Clean circles')
    axs[0, 1].set_title('Diagrams of clean circles')
    axs[0, 2].set_title('Barycenter')
    axs[0, 4].set_title('Diagrams of noisy circles')
