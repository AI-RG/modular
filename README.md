# Learning modularity from scratch
An investigation of modularity-inducing regularization terms. The goal is to learn modular network structure from scratch.

## Background
In fuctional, real-world networks ranging from gene regulation networks to human brain connectomes, modularity plays an important role. Informally, modularity is a property of networks (graphs) whereby nodes can be grouped into regions (modules) that have many more interconnections (in-module) than intraconnections (out-of-module). There have been a large number investigations into the evolutionary and dynamical processes that give rise to modularity, as well as what benefits modularity can confer to functional networks. These benefits include robustness to perturbations, faster evolvability/adaptibility, and dynamical criticality (along with its associated information-processing benefits). Moreover, small modules or cliques of neurons have been shown to be a crucial ingedient in short-term memory, at least in a tractable, simplified model.

Given the recognized utility of modularity in real-world networks, it is natural to ask whether modularity can aid artificial neural networks as well. Indeed, many existing and performant network architectures can be interpreted as imposing a strong prior of modularity. All weight-sharing architectures vastly increase modularity over a randomly connected network; the layer paradigm of deep learning also leads to much higher network modularity than a standard null model, e.g. randomly connected neurons across all layers. Here, the modules are pairs of adjacent layers, which have many more interconnections per neuron (they are fully connected) than out-of-module connections per neuron (this is true for a network of more than three layers). Recent work has investigated more explicitly modular structues: Neural Programmer-Interpreters learn subroutines; in hierarchical RL, primitive motor-policies or options are learned which can then be recombined in a modular way. In each of these cases, a modular architecture is presupposed and hard-wired. This study investigates the following questions: can a modular architecture be learned from scratch? And if so, can this learned modularity improve network performance in some regards? 

## Training procedure

There is nothing to prevent the neural network practitioner from directly optimizing network modularity. There are many ways to achieve this. One study of modularity used an MLP with tanh-nonlinearity and penalized the sum of squared connection lengths in addition to optimizing classification performance; this study imagined that the nodes of the network were literally embedded in three dimensions, so that the connection length cost was literally due to e.g. the cost of producing the wiring material over a longer distance. This model is appropriate for investigating neurons in the human brain and may also be well suited to processing 2-d images in general, but places too strong an assumption on the correlation of neighboring inputs to be viable for all ANNs. For simplicity and interpretibility, we choose to optimize using a weight-decay-like criterion in inspired by []. A straightforward, local way to increase modularity as a network evolves is to sample at each time step a group of three nodes (neurons) and increase the weight of the smallest-weight edge. In more detail, the algorithm does the following:
- sample a directed edge (uniformly)
- sample a child (with probability proportional to the edge weight)
- increase the weight of the connection between the original parent and the final child, thus forming a three-node clique
We cannot directly apply this procedure to ANNs for a few reasons. The most obvious of these is that in a layered neural network, there are no three-node cliques--the smallest loop in an ANN consists of four nodes. Thus our algorithm emulates the above but by reinforcing four-node cliques. Moreover, for simplicity of integration with traditional learning, we implement this as a differentiable loss term that is essentially a different kind of regularization.

Concretely, we add the term:

Concretely, we consider the following regularization term:
\begin{eqnarray}
L_{mod} & = & -\sum_{L} \sum_{ijkl}W^{(L)}_{ij}W^{(L)}_{ik}W^{(L+1)}_{jl}W^{(L+1)}_{jl} \nonumber 
\end{eqnarray}
Because this term tends to increase weight values, we must balance it with an appriate weight decay term:
\begin{eqnarray}
L_{reg} & = & -\left({L_{mod}}\right)^{\frac{1}{2}} + C\sum_{L} \sum_{ij}\left(W^{(L)}_{ij}\right)^4 \nonumber
\end{eqnarray}
This regularization has the effect of suppressing values of weights that become too large, and due to the difference in powers, no weight can grow unbounded in magnitude.

This penalty is designed to work with stochastic gradient descent in such a way that each update step increases the values of the lowest weights in four-node loops:
\begin{eqnarray}
\delta W^{(L)}_{ij} \propto \frac{\partial}{\partial W^{(L)}_{ij}}L_{mod} & = & \sum_{kl}W^{(L)}_{ik}W^{(L+1)}_{jl}W^{(L+1)}_{jl} \nonumber
\end{eqnarray}
The larger the three other edges are in a loop, the more the final edge will be increased. Moreover, this particular choice of regularization has the effect of increasing this weight most when the weights among the other edges are evenly distributed.




## Results and future directions


## Details
