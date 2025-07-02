You should implement Layer Normalization using python and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


Batch Normalization has limitations: need to maintain running means;tricky for RNNs; we don't need different normalizations for each step;doesn't work with small batch sizes; large NLP models are usually trained with small batch sizes;need to compute means and variances across devices in distributed training.
Layer normalization is a simpler normalization method that works on a wider range of settings. Layer normalization transforms the inputs to have zero mean and unit variance across the features. Batch normalization fixes the zero mean and unit variance for each element, while layer normalization does it for each batch across all elements.
The module should be named LayerNorm.
The LayerNorm module normalizes (layer normalization) the input $X$ as follows:
When input $X \in \mathbb{R}^{B \times C}$ is a batch of embeddings, where $B$ is the batch size and $C$ is the number of features. $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$.
$$
\text{LN}(X) = \gamma
\frac{X - \underset{C}{\mathbb{E}}[X]}{\sqrt{\underset{C}{Var}[X] + \epsilon}}
+ \beta
$$
When input $X \in \mathbb{R}^{L \times B \times C}$ is a batch of a sequence of embeddings, where $B$ is the batch size, $C$ is the number of channels, $L$ is the length of the sequence. $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$.
$$
\text{LN}(X) = \gamma
\frac{X - \underset{C}{\mathbb{E}}[X]}{\sqrt{\underset{C}{Var}[X] + \epsilon}}
+ \beta
$$
When input $X \in \mathbb{R}^{B \times C \times H \times W}$ is a batch of image representations, where $B$ is the batch size, $C$ is the number of channels, $H$ is the height and $W$ is the width. This is not a widely used scenario. $\gamma \in \mathbb{R}^{C \times H \times W}$ and $\beta \in \mathbb{R}^{C \times H \times W}$.
$$
\text{LN}(X) = \gamma
\frac{X - \underset{C, H, W}{\mathbb{E}}[X]}{\sqrt{\underset{C, H, W}{Var}[X] + \epsilon}}
+ \beta
$$

The init function needs to include the following parameters:
normalized_shape: the shape of the elements (except the batch). The input should be $X \in \mathbb{R}^{* \times S[0] \times S[1] \times ... \times S[n]}$;
eps: hyperparameter $\epsilon$, used in $\sqrt{Var[x^{(k)}] + \epsilon}$ for numerical stability. The default value is $10^{-5}$;
elementwise_affine: whether to scale and shift the normalized value. The default value is True;

The module needs to include at least the following functions:
1. forward: function to normalize the input. It should include the following parameters:
   x: a tensor of shape [*, S[0], S[1], ..., S[n]]. * could be any number of dimensions. For example, in an NLP task this will be [seq_len, batch_size, features].
   The return of forward function includes:
   x_norm: the normalization result of input x.

You just need to implement the algorithm module; no need to provide corresponding examples.