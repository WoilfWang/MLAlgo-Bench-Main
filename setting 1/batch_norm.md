You should implement Batch Normalization using python and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


The module should be named BatchNorm.

The BatchNorm module is a batch normalization layer that normalizes the input $X$ as follows:

When input $X \in \mathbb{R}^{B \times C \times H \times W}$ is a batch of image representations, where $B$ is the batch size, $C$ is the number of channels, $H$ is the height and $W$ is the width. $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$.
$$
\text{BN}(X) = \gamma
\frac{X - \underset{B, H, W}{\mathbb{E}}[X]}{\sqrt{\underset{B, H, W}{Var}[X] + \epsilon}}
+ \beta
$$
When input $X \in \mathbb{R}^{B \times C}$ is a batch of embeddings, where $B$ is the batch size and $C$ is the number of features. $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$.
$$
\text{BN}(X) = \gamma
\frac{X - \underset{B}{\mathbb{E}}[X]}{\sqrt{\underset{B}{Var}[X] + \epsilon}}
+ \beta
$$
When input $X \in \mathbb{R}^{B \times C \times L}$ is a batch of a sequence embeddings, where $B$ is the batch size, $C$ is the number of features, and $L$ is the length of the sequence. $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$​.
$$
\text{BN}(X) = \gamma
\frac{X - \underset{B, L}{\mathbb{E}}[X]}{\sqrt{\underset{B, L}{Var}[X] + \epsilon}}
+ \beta
$$

The init function needs to include the following parameters:
channels: the number of features in the input;
eps: hyperparameter $\epsilon$, used in $\sqrt{Var[x^{(k)}] + \epsilon}$ for numerical stability. The default value is $10^{-5}$;
momentum: the momentum in taking the exponential moving average. The default value is 0.1;
affine: whether to scale and shift the normalized value. The default value is True;
track_running_stats: whether to calculate the moving averages or mean and variance. The default value is True;

The module needs to include at least the following functions:
1. forward: function to normalize the input. It should include the following parameters:
   x: a tensor of shape [batch_size, channels, *]. * denotes any number (possibly 0) of dimensions. For example, in an image (2D) convolution this will be [batch_size, channels, height, width].
   The return of forward function includes:
   x_norm: the normalization result of input. It has the same shape with input x.

You just need to implement the algorithm module; no need to provide corresponding examples.