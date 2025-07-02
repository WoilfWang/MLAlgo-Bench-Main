You should implement Capsule Networks using Python and PyTorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


Capsule network is a neural network architecture that embeds features as capsules and routes them with a voting mechanism to next layer of capsules. 
The Capsule Networks algorithm aims to overcome the limitations of traditional convolutional neural networks (CNNs) in handling hierarchical spatial relationships. It introduces "capsules," which are groups of neurons whose activity vectors represent various instantiation parameters of a specific type of entity. These capsules are arranged in layers, with each layer capturing different levels of abstraction.The core idea is the dynamic routing algorithm, where lower-level capsules predict the poses (instantiation parameters) of higher-level capsules. These predictions are based on the agreement between the lower-level capsules' outputs and the higher-level capsules' inputs.The algorithm iteratively refines these predictions through routing by agreement, ensuring that higher-level capsules receive inputs from lower-level capsules whose outputs are in agreement with the predictions. This dynamic routing mechanism facilitates the learning of hierarchical relationships between features, enabling better generalization and robustness.

The model needs to include at least the following classes:

1.class Squash
This is **squashing** function from paper, given by equation $(1)$.
$$\mathbf{v}_j = \frac{{\lVert \mathbf{s}_j \rVert}^2}{1 + {\lVert \mathbf{s}_j \rVert}^2}\frac{\mathbf{s}_j}{\lVert \mathbf{s}_j \rVert}$$
$\frac{\mathbf{s}_j}{\lVert \mathbf{s}_j \rVert}$
normalizes the length of all the capsules, whilst $\frac{{\lVert \mathbf{s}_j \rVert}^2}{1 + {\lVert \mathbf{s}_j \rVert}^2}$ shrinks the capsules that have a length smaller than one .
The module should be named Squash.
The init function needs to include the following parameters:
epsilon: A small value added to the denominator to prevent division by zero. The default value is 1e-8.
The forward function needs to include the following parameters:
s: Tensor holding the input data with shape [batch_size, n_capsules, n_features].
The return of forward function includes:
v: Tensor of squashed output capsules with shape [batch_size, n_capsules, n_features]. $\mathbf{v}_j = \frac{{\lVert \mathbf{s}_j \rVert}^2}{1 + {\lVert \mathbf{s}_j \rVert}^2} \frac{\mathbf{s}_j}{\sqrt{{\lVert \mathbf{s}_j \rVert}^2 + \epsilon}}$

2.class Router
This is the routing mechanism described in the paper. You can use multiple routing layers in your models. This combines calculating $\mathbf{s}_j$ for this layer and the routing algorithm described in *Procedure 1*.
The module should be named Router.
The init function needs to include the following parameters:
in_caps: Number of capsules from the previous layer.
out_caps: Number of capsules for the current layer.
in_d: Number of features per capsule from the previous layer.
out_d: Number of features per capsule for the current layer.
iterations: Number of routing iterations.
The forward function needs to include the following parameters:
u: Tensor containing capsules from the lower layer with shape [batch_size, n_capsules, n_features].
The return of forward function includes:
v: Tensor of squashed output capsules with shape [batch_size, n_capsules, n_features].

3.class MarginLoss
A separate margin loss is used for each output capsule and the total loss is the sum of them. The length of each output capsule is the probability that class is present in the input. Loss for each output capsule or class $k$ is, $$\mathcal{L}_k = T_k \max(0, m^{+} - \lVert\mathbf{v}_k\rVert)^2 +  \lambda (1 - T_k) \max(0, \lVert\mathbf{v}_k\rVert - m^{-})^2$$.  
$T_k$ is $1$ if the class $k$ is present and $0$ otherwise. The first component of the loss is $0$ when the class is not present, and the second component is $0$ if the class is present. The $\max(0, x)$ is used to avoid predictions going to extremes. $m^{+}$ is set to be $0.9$ and $m^{-}$ to be $0.1$ in the paper.  The $\lambda$ down-weighting is used to stop the length of all capsules from falling during the initial phase of training.
The module should be named MarginLoss.
The init function needs to include the following parameters:
n_labels: Number of labels.
lambda_: Weight parameter for down-weighting. The default value is 0.5;
m_positive: Margin for positive samples. The default value is 0.9;
m_negative: Margin for negative samples. The default value is 0.1.
The forward function needs to include the following parameters:
v: Tensor of squashed output capsules with shape [batch_size, n_labels, n_features].
labels: Tensor containing the labels with shape [batch_size].
The return of forward function includes:
loss: Scalar tensor representing the total margin loss.


You just need to implement the algorithm module; no need to provide corresponding examples.