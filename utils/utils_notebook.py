import matplotlib.pyplot as plt
import numpy as np

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