# Topological Uncertainty

This repository provides code related to the paper _[Topological Uncertainty: Monitoring Trained Neural Networks 
through Persistence of Activation Graphs](https://arxiv.org/pdf/2105.04404.pdf)_. 
It aims at being integrated to the [Gudhi](https://gudhi.inria.Fr) library in the future.

**Note:** this repository is still in a preliminary state. Do not hesitate to report any issue you come accross or 
suggest options you may like to see in the future.

## Summary

In a nutshell, the idea is to extract _topological information_ from a sample of weighted graphs (with fixed 
combinatorics) and then aggregate this topological information through a _Fréchet mean_ (a.k.a barycenter). 
Given a new observation (graph), we can compute its (topological) distance to this mean and use this quantity, called
_Topological Uncertainty_, as a way to detect unusual weight distributions.

In the aforementioned paper, we used this technique in the context of monitoring neural networks (as a way to 
detect Out-of-Distribution samples and similar anomalies), but this code can be used in a more general setting. 

**Note:** in the context of neural networks, current implementation only handles `tensorflow 2` sequential 
networks and fully-connected layers (seen as bipartite graphs): 
your (sequential) network can contain convolutional layers etc., but topological information is only extracted 
from the folly-connected layers for now. 
Further implementations (in particular with `Torch`) will come in the future.

## Get started

### Dependencies
This code was developed and tested with Ubuntu 20.04 (it is likely that other versions/OS work as well) 
and `Python 3.8` # (note: `tensorflow 2.4` is not compatible `Python 3.9`), 
using the following python packages (stared packages are required, others are optional):

- `numpy 1.20` *
- `gudhi 3.3.0` *  (used to store activation graphs and compute MST/diagrams on top of them)
- `tensorflow 2.4.1`

Note that `tensorflow` is only needed if you want to manipulate fully connected neural networks.

## Play with the tutorial

We provide a tutorial to play quickly with these notions and see how you can use them for your own purpose. 
It shows how you can get TU  
- directly from a set of graphs saved as either `networkx.graphs`, `scipy.sparse` adjacency matrices or simply
`numpy` adjacency matrices.
- From a trained (`tensorflow`, sequential) neural network.
 

Finally, an experiment (reproducing a result in the reference paper) using a network trained on the 
`MUTAG` and `COX2` [datasets](www.graphlearning.io) is also provided 
at the end of the notebook. 
For the sake of simplicity, these datasets are provided in the `./datesets/` folder.

### Notebook additional dependencies:

Along with the aforementioned dependencies, the notebook makes use of the following packages:
- `matplotlib.pyplot 1.19.2`
- `pandas 1.1.3`  (used to store features computed from MUTAG and COX2)


### Run the notebook:

We suggest to use a fresh conda env. For this, you can start by typing in a terminal 

```
conda create -n test-TU python=3.8 jupyterlab
conda activate test-TU
```
You can then install the dependecies, for instance using
```
conda install pandas tensorflow
conda install -c conda-forge gudhi matplotlib
```