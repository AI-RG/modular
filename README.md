# Learning modularity from scratch
An investigation of modularity-inducing regularization terms. The goal is to learn modular network structure from scratch.

## Background
In fuctional, real-world networks ranging from gene regulation networks to human brain connectomes, modularity plays an important role. Informally, modularity is a property of networks (graphs) whereby nodes can be grouped into regions (modules) that have many more interconnections (in-module) than intraconnections (out-of-module). There have been a large number investigations into the evolutionary and dynamical processes that give rise to modularity, as well as what benefits modularity can confer to functional networks. These benefits include robustness to perturbations, faster evolvability/adaptibility, and dynamical criticality (along with its associated information-processing benefits). Moreover, small modules or cliques of neurons have been shown to be a crucial ingedient in short-term memory, at least in a tractable, simplified model.

Given the recognized utility of modularity in real-world networks, it is natural to ask whether modularity can aid artificial neural networks as well. Indeed, many existing and performant network architectures can be interpreted as imposing a strong prior of modularity. All weight-sharing architectures vastly increase modularity over a randomly connected network; the layer paradigm of deep learning also leads to much higher network modularity than a standard null model, e.g. randomly connected neurons across all layers. Here, the modules are pairs of adjacent layers, which have many more interconnections per neuron (they are fully connected) than out-of-module connections per neuron (this is true for a network of more than three layers). Recent work has investigated more explicitly modular structues: Neural Programmer-Interpreters learn subroutines; in hierarchical RL, primitive motor-policies or options are learned which can then be recombined in a modular way. In each of these cases, modularity is presupposed. This study investigates the following questions: can a modular architecture be learned from scratch? And if so, can this learned modularity improve network performance in some regards? 

## Training procedure

In order to 


## Results and future experiments


## Details
