You should implement Batch-Channel Normalization using python and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


Batch-Channel Normalization performs batch normalization followed by a channel normalization (similar to a Group Normalization). When the batch size is small a running mean and variance is used for batch normalization.
Batch-Channel Normalization first performs a batch normalization which is either normal batch norm or a batch norm with estimated mean and variance (exponential mean/variance over multiple batches). Then a channel normalization performed.

The module should be named BatchChannelNorm.

The init function needs to include the following parameters:
channels: the number of the features in the input;
groups: the number of groups the features are divided into;
eps: hyperparameter $\epsilon$, used in $\sqrt{Var[x^{(k)}] + \epsilon}$ for numerical stability. The default value is $10^{-5}$;
momentum: the momentum in taking the exponential moving average. The default value is 0.1;
estimate: whether to use running mean and variance for batch norm. The default value is True;

The module needs to include at least the following functions:

1. forward: function to normalize the input. It should include the following parameters:
   x: a tensor of shape [batch_size, channels, *]. * denotes any number of (possibly 0) dimensions. For example, in an image (2D) convolution this will be [batch_size, channels, height, width].
   The return of forward function includes:
   x_norm: the normalization result of input. It has the same shape with input x.
You just need to implement the algorithm module; no need to provide corresponding examples.