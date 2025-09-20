You should implement Group Normalization using python and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


Batch Normalization works well for large enough batch sizes but not well for small batch sizes, because it normalizes over the batch. Training large models with large batch sizes is not possible due to the memory capacity of the devices.
Group Normalization normalizes a set of features together as a group. This is based on the observation that classical features such as the scale-invariant feature transform (SIFT) and the histogram of oriented gradients (HOG) are group-wise features. Group Normalization divides feature channels into groups and then separately normalizes all channels within each group.
All normalization layers can be defined by the following computation.
$$
\hat{x}_i = \frac{1}{\sigma_i} (x_i - \mu_i)
$$
where $x$ is the tensor representing the batch, and $i$ is the index of a single value. For instance, when it's 2D images $i = (i_N, i_C, i_H, i_W)$ is a 4-d vector for indexing image within batch, feature channel, vertical coordinate and horizontal coordinate. $\mu_i$ and $\sigma_i$​ are mean and standard deviation.
$$
\begin{align}
\mu_i &= \frac{1}{m} \sum_{k \in \mathcal{S}_i} x_k \\
\sigma_i  &= \sqrt{\frac{1}{m} \sum_{k \in \mathcal{S}_i} (x_k - \mu_i)^2 + \epsilon}
\end{align}
$$
$\mathcal{S}_i$ is the set of indexes across which the mean and standard deviation are calculated for index $i$. $m$ is the size of the set $\mathcal{S}_i$ which is the same for all $i$.

The definition of $\mathcal{S}_i$ is different for different normalization methods. Group normalization normalizes values of the same sample and the same group of channels together. $\mathcal{S}_i$ in group normalization computes as:

$$
\mathcal{S}_i = \{k | k_N = i_N,
 \bigg \lfloor \frac{k_C}{C/G} \bigg \rfloor = \bigg \lfloor \frac{i_C}{C/G} \bigg \rfloor\}
$$
where $G$ is the number of groups and $C$ is the number of channels.
The module should be named GroupNorm.

The GroupNorm module normalizes (group normalization) the input $X$​​ as follows:

1. Calculate the mean across last dimension, i.e. the means for each sample and channel group $\mathbb{E}[x_{(i_N, i_G)}]$.
2. Calculate the squared mean across last dimension, i.e. the means for each sample and channel group $\mathbb{E}[x^2_{(i_N, i_G)}]$.
3. Variance for each sample and feature group computes as: $Var[x_{(i_N, i_G)}] = \mathbb{E}[x^2_{(i_N, i_G)}] - \mathbb{E}[x_{(i_N, i_G)}]^2$.
4. Normalize the $x$ as: $\hat{x}_{(i_N, i_G)} =
   \frac{x_{(i_N, i_G)} - \mathbb{E}[x_{(i_N, i_G)}]}{\sqrt{Var[x_{(i_N, i_G)}] + \epsilon}}$.
5. Scale and shift channel-wise: $y_{i_C} =\gamma_{i_C} \hat{x}_{i_C} + \beta_{i_C}$.
The init function needs to include the following parameters:
groups: the number of groups the features are divided into;
channels: the number of the features in the input;
eps: hyperparameter $\epsilon$, used in $\sqrt{Var[x^{(k)}] + \epsilon}$ for numerical stability. The default value is $10^{-5}$;
affine: whether to scale and shift the normalized value. The default value is True;
The module needs to include at least the following functions:

1. forward: function to normalize the input. It should include the following parameters:
   x: a tensor of shape [batch_size, channels, *]. * denotes any number of (possibly 0) dimensions. For example, in an image (2D) convolution this will be [batch_size, channels, height, width].
   The return of forward function includes:
   x_norm: the normalization result of input. It has the same shape with input x.
You just need to implement the algorithm module; no need to provide corresponding examples.