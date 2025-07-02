You should implement Instance Normalization using python and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


Style Transfer is a technique in computer vision and graphics that involves generating a new image by combining the content of one image with the style of another image. The goal of style transfer is to create an image that preserves the content of the original image while applying the visual style of another image.
Instance normalization was introduced to improve style transfer. It is based on the observation that stylization should not depend on the contrast of the content image. A simple version of contrast normalization is given by:
$$
y_{t,i,j,k} = \frac{x_{t,i,j,k}}{\sum_{l=1}^H \sum_{m=1}^W x_{t,i,l,m}}
$$
where $x$ is a batch of images with dimensions image index $t$, feature channel $i$, and spatial position $j, k$.

Since it's hard for a convolutional network to learn "contrast normalization", instance normalization does that.

The module should be named InstanceNorm.
The InstanceNorm module normalizes (instance normalization) the input $X$​ as follows:
When input $X \in \mathbb{R}^{B \times C \times H \times W}$ is a batch of image representations, where $B$ is the batch size, $C$ is the number of channels, $H$ is the height and $W$ is the width. $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$. The affine transformation with $gamma$ and $beta$ are optional.
$$
\text{IN}(X) = \gamma
\frac{X - \underset{H, W}{\mathbb{E}}[X]}{\sqrt{\underset{H, W}{Var}[X] + \epsilon}}
+ \beta
$$
The init function needs to include the following parameters:
channels: the number of the features in the input;
eps: hyperparameter $\epsilon$, used in $\sqrt{Var[x^{(k)}] + \epsilon}$ for numerical stability. The default value is $10^{-5}$;
affine: whether to scale and shift the normalized value. The default value is True;
The module needs to include at least the following functions:
1. forward: function to normalize the input. It should include the following parameters:
   x: a tensor of shape [batch_size, channels, *]. * denotes any number of (possibly 0) dimensions. For example, in an image (2D) convolution this will be [batch_size, channels, height, width].
   The return of forward function includes:
   x_norm: the normalization result of input. It has the same shape with input x.

You just need to implement the algorithm module; no need to provide corresponding examples.