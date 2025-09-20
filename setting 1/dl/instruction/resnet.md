You should implement Deep Residual Learning for Image Recognition (ResNet) using Python and PyTorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


ResNets train layers as residual functions to overcome the degradation problem. The degradation problem is the accuracy of deep neural networks degrading when the number of layers becomes very high. The accuracy increases as the number of layers increase, then saturates, and then starts to degrade.
Residual Learning
If $\mathcal{H}(x)$ is the mapping that needs to be learned by a few layers, they train the residual function $\mathcal{F}(x) = \mathcal{H}(x) - x$ instead. And the original function becomes $\mathcal{F}(x) + x$. In this case, learning identity mapping for $\mathcal{H}(x)$ is equivalent to learning $\mathcal{F}(x)$ to be 0, which is easier to learn. In the parameterized form this can be written as, $\mathcal{F}(x, \{W_i\}) + x$ and when the feature map sizes of$ \mathcal{F}(x, {W_i})$ and $x$ are different the paper suggests doing a linear projection, with learned weights $W_s$.
$\mathcal{F}(x, \{W_i\}) + W_s x$, $\mathcal{F}$ should have more than one layer, otherwise the sum $\mathcal{F}(x, \{W_i\}) + W_s x$ also won't have non-linearities and will be like a linear layer.

The model needs to include at least the following classes:

1.class ResNetBase.
This is a the base of the resnet model without the final linear layer and softmax for classification.
The resnet is made of stacked residual blocks or bottleneck residual blocks. The feature map size is halved after a few blocks with a block of stride length 2. The number of channels is increased when the feature map size is reduced. Finally the feature map is average pooled to get a vector representation.
The module should be named ResNetBase.
The init function needs to include the following parameters:
n_blocks: is a list of of number of blocks for each feature map size;
n_channels: is the number of channels for each feature map sizel;
Bottlenecks: is the number of channels the bottlenecks. If this is `None`, [residual  blocks](#residual_block) are used;
img_channels: is the number of channels in the input. The default value is 3;
first_kernel_size: is the kernel size of the initial convolution layer. The default value is 7.
The forward function(the forward pass of the network, processing the input tensor through the initial convolutional layer, batch normalization, and residual blocks, followed by global average pooling.)needs to include the following parameters:
x (torch.Tensor): Input tensor with shape [batch_size, img_channels, height, width].
The return of forward function includes:
torch.Tensor with shape [batch_size, channels].

You just need to implement the algorithm module; no need to provide corresponding examples.