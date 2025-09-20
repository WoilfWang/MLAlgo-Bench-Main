You should implement StyleGAN2. using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 

We'll first introduce the three papers at a high level.

## Generative Adversarial Networks

Generative adversarial networks have two components; the generator and the discriminator.
The generator network takes a random latent vector ($z \in \mathcal{Z}$)
 and tries to generate a realistic image.
The discriminator network tries to differentiate the real images from generated images.
When we train the two networks together the generator starts generating images indistinguishable from real images.

## Progressive GAN

Progressive GAN generates high-resolution images ($1080 \times 1080$) of size.
It does so by *progressively* increasing the image size.
First, it trains a network that produces a $4 \times 4$ image, then $8 \times 8$ ,
 then an $16 \times 16$  image, and so on up to the desired image resolution.

At each resolution, the generator network produces an image in latent space which is converted into RGB,
with a $1 \times 1$  convolution.
When we progress from a lower resolution to a higher resolution
 (say from $4 \times 4$  to $8 \times 8$ ) we scale the latent image by $2\times$
 and add a new block (two $3 \times 3$  convolution layers)
 and a new $1 \times 1$  layer to get RGB.
The transition is done smoothly by adding a residual connection to
 the $2\times$ scaled $4 \times 4$  RGB image.
The weight of this residual connection is slowly reduced, to let the new block take over.

The discriminator is a mirror image of the generator network.
The progressive growth of the discriminator is done similarly.

![progressive_gan.svg](progressive_gan.svg)

---*$2\times$ and $0.5\times$ denote feature map resolution scaling and scaling.
$4\times4$, $8\times4$, ... denote feature map resolution at the generator or discriminator block.
Each discriminator and generator block consists of 2 convolution layers with leaky ReLU activations.*---

They use **minibatch standard deviation** to increase variation and
 **equalized learning rate** which we discussed below in the implementation.
They also use **pixel-wise normalization** where at each pixel the feature vector is normalized.
They apply this to all the convolution layer outputs (except RGB).


## StyleGAN

StyleGAN improves the generator of Progressive GAN keeping the discriminator architecture the same.

#### Mapping Network

It maps the random latent vector ($z \in \mathcal{Z}$)
 into a different latent space ($w \in \mathcal{W}$),
 with an 8-layer neural network.
This gives an intermediate latent space $\mathcal{W}$
where the factors of variations are more linear (disentangled).

#### AdaIN

Then $w$ is transformed into two vectors (**styles**) per layer,
 $i$, $y_i = (y_{s,i}, y_{b,i}) = f_{A_i}(w)$ and used for scaling and shifting (biasing)
 in each layer with $\text{AdaIN}$ operator (normalize and scale):
$$\text{AdaIN}(x_i, y_i) = y_{s, i} \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}$$

#### Style Mixing

To prevent the generator from assuming adjacent styles are correlated,
 they randomly use different styles for different blocks.
That is, they sample two latent vectors $(z_1, z_2)$ and corresponding $(w_1, w_2)$ and
 use $w_1$ based styles for some blocks and $w_2$ based styles for some blacks randomly.

#### Stochastic Variation

Noise is made available to each block which helps the generator create more realistic images.
Noise is scaled per channel by a learned weight.

#### Bilinear Up and Down Sampling

All the up and down-sampling operations are accompanied by bilinear smoothing.

![style_gan.svg](style_gan.svg)

---*$A$ denotes a linear layer.
$B$ denotes a broadcast and scaling operation (noise is a single channel).
StyleGAN also uses progressive growing like Progressive GAN.*---

## StyleGAN 2

StyleGAN 2 changes both the generator and the discriminator of StyleGAN.

#### Weight Modulation and Demodulation

They remove the $\text{AdaIN}$ operator and replace it with
 the weight modulation and demodulation step.
This is supposed to improve what they call droplet artifacts that are present in generated images,
 which are caused by the normalization in $\text{AdaIN}$ operator.
Style vector per layer is calculated from $w_i \in \mathcal{W}$ as $s_i = f_{A_i}(w_i)$.

Then the convolution weights $w$ are modulated as follows.
($w$ here on refers to weights not intermediate latent space,
 we are sticking to the same notation as the paper.)

$$w'_{i, j, k} = s_i \cdot w_{i, j, k}$$
Then it's demodulated by normalizing,
$$w''_{i,j,k} = \frac{w'_{i,j,k}}{\sqrt{\sum_{i,k}{w'_{i, j, k}}^2 + \epsilon}}$$
where $i$ is the input channel, $j$ is the output channel, and $k$ is the kernel index.

#### Path Length Regularization

Path length regularization encourages a fixed-size step in $\mathcal{W}$ to result in a non-zero,
 fixed-magnitude change in the generated image.

#### No Progressive Growing

StyleGAN2 uses residual connections (with down-sampling) in the discriminator and skip connections
 in the generator with up-sampling
  (the RGB outputs from each layer are added - no residual connections in feature maps).
They show that with experiments that the contribution of low-resolution layers is higher
 at beginning of the training and then high-resolution layers take over.

You should implement five modules: Discriminator, Generator, MappingNetwork, Gradient Penalty, Path Length Penalty

1. MappingNetwork
This is an MLP with 8 linear layers.
The mapping network maps the latent vector $z \in \mathcal{W}$ to an intermediate latent space $w \in \mathcal{W}$. 
$\mathcal{W}$ space will be disentangled from the image space where the factors of variation become more linear.

The module should be named MappingNetwork.
The init function needs to include the following parameters:
features: `features` is the number of features in $z$ and $w$;
n_layers: `n_layers` is the number of layers in the mapping network.

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
z: z is the input tensor.
It should return:
result: the result of MappingNetwork.

2. Generator
StyleGAN2 Generator. 
1.	Initial Constant Input:
	•	The network begins with a learned constant tensor of shape 512×4×4, which acts as the starting feature map for the generation process.
	2.	Style Injection:
	•	A latent vector (A) is processed through a Modulation (Mod) operation, which adjusts the convolution weights based on the style vector.
	•	The resulting modulated weights undergo a Demodulation (Demod) process to normalize the convolutional operation.
	3.	3×3 Convolution:
	•	A 3×3 Convolution is applied to the feature map. This operation, coupled with the modulated and demodulated weights, introduces spatial patterns influenced by the style.
	4.	Bias Addition:
	•	A bias (B) is added to the feature map after the convolution, enhancing flexibility in the representation.
	5.	RGB Output:
	•	The processed feature map is converted to an RGB image using a toRGB layer, which maps the internal representation to a 3-channel output (one per color).
	6.	Upsampling and Layer Stacking:
	•	After generating the first 4×4 resolution, the feature map is upsampled to increase spatial resolution (e.g., from 4×4 to 8×8). This process repeats for progressively larger resolutions in the network.
	•	Each upsampled feature map undergoes the same set of operations (modulation, demodulation, convolution, bias addition, and toRGB mapping).
	7.	Progressive Synthesis:
	•	The intermediate RGB outputs are added together during the upsampling process, contributing to the final high-resolution image.
	8.	Repeatable Structure:
	•	The structure repeats for higher resolutions, with more layers performing similar steps while increasing the spatial size of the output image.

This architecture is characteristic of StyleGAN2’s generator, where the style vector (A) controls the visual attributes of the generated image at different resolutions, and progressive refinement ensures that fine details are added to the image as resolution increases.

The generator starts with a learned constant. Then it has a series of blocks. The feature map resolution is doubled at each block Each block outputs an RGB image and they are scaled up and summed to get the final RGB image.

The module should be named Generator.
The init function needs to include the following parameters:
log_resolution: `log_resolution`(int) is the $\log_2$ of image resolution;
d_latent: `d_latent`(int) is the dimensionality of $w$;
n_features: `n_features` number of features in the convolution layer at the highest resolution (final block). The default value is 32;
max_features: `max_features` maximum number of features in any generator block. The default value is 512.
It contains the class variable n_blocks, the number of generator blocks.

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
w: `w`(torch.Tensor) is $w$. In order to mix-styles (use different $w$ for different layers), we provide a separate $w$ for each [generator block](#generator_block). It has shape `[n_blocks, batch_size, d_latent]`.
input_noise: `input_noise`(List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]) is the noise for each block. It's a list of pairs of noise sensors because each block (except the initial) has two noise inputs after each convolution layer (see the diagram).
It should return:
rgb_img: the final RGB image.

3.Discriminator
StyleGAN 2 Discriminator.
Discriminator first transforms the image to a feature map of the same resolution and then runs it through a series of blocks with residual connections. The resolution is down-sampled by 2× at each block while doubling the number of features.

	1.	Input Layer (fromRGB):
The network starts by processing the RGB input image of size 1024 \times 1024. The input undergoes transformations to extract initial features.
	2.	Feature Extraction (Convolution and Downsampling):
	•	The first layer applies a 3 \times 3 convolutional filter followed by a downsampling operation, reducing the spatial dimensions of the feature maps.
	•	A parallel path applies a 1 \times 1 convolution, which compresses the features along the channel dimension without altering the spatial resolution.
	•	The outputs of the two paths are added together to combine the processed features.
	3.	Progressive Downsampling and Convolution:
	•	As the feature maps progress through the network, their resolution is reduced iteratively by the downsampling operation after each 3 \times 3 convolution.
	•	These layers refine the features by focusing on relevant patterns while progressively reducing the spatial dimensions.
	4.	Transition to Smaller Resolutions (8x8 and Below):
	•	When the feature maps are reduced to 8 \times 8, similar convolutional and downsampling operations are applied.
	•	The network uses skip connections, where feature maps from earlier layers are directly connected to later layers, facilitating gradient flow and retaining fine-grained details.
	5.	Final Processing Layers:
	•	At 4 \times 4, a Minibatch Standard Deviation layer is applied, likely to regularize the network by introducing batch-level statistics into the feature representation.
	•	A 3 \times 3 convolution follows to further process the features.
	6.	Flattening and Classification:
	•	At 2 \times 2, the feature maps are flattened into a 1D vector and passed to a fully connected layer for classification.

The module should be named Discriminator.
The init function needs to include the following parameters:
log_resolution: `log_resolution` is the $\log_2$ of image resolution;
n_features: `n_features` number of features in the convolution layer at the highest resolution (first block). The default value is 64;
max_features: `max_features` maximum number of features in any generator block. The default value is 512. 

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
x: `x`(torch.Tensor) is the input image of shape `[batch_size, 3, height, width]`.
It should return:
score: the classification score.

4. Gradient Penalty
This is the $R_1$ regularization penality.
$$R_1(\psi) = \frac{\gamma}{2} \mathbb{E}_{p_\mathcal{D}(x)} \Big[\Vert \nabla_x D_\psi(x)^2 \Vert\Big]$$
That is we try to reduce the L2 norm of gradients of the discriminator with respect to images, for real images ($P_\mathcal{D}$).

The module should be named GradientPenalty.
The initialization function does not contain any input parameters.

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
x: `x`(torch.Tensor) is $x \sim \mathcal{D}$;
d: `d`(torch.Tensor) is $D(x)$.
It should return:
loss: Return the loss $\Vert \nabla_x D_\psi(x)^2 \Vert$.

5. Path Length Penalty

This regularization encourages a fixed-size step in $w$ to result in a fixed-magnitude change in the image.
$$\mathbb{E}_{w \sim f(z), y \sim \mathcal{N}(0, \mathbf{I})} \Big(\Vert \mathbf{J}^\top_{w} y \Vert_2 - a \Big)^2$$
where $\mathbf{J}_w$ is the Jacobian $\mathbf{J}_w = \frac{\partial g}{\partial w}$, $w$ are sampled from $w \in \mathcal{W}$ from the mapping network, and $y$ are images with noise $\mathcal{N}(0, \mathbf{I})$.
$a$ is the exponential moving average of $\Vert \mathbf{J}^\top_{w} y \Vert_2$ as the training progresses.
$\mathbf{J}^\top_{w} y$ is calculated without explicitly calculating the Jacobian using
$$\mathbf{J}^\top_{w} y = \nabla_w \big(g(w) \cdot y \big)$$

The module should be named PathLengthPenalty.
The init function needs to include the following parameters:
beta: `beta` is the constant $\beta$ used to calculate the exponential moving average $a$.

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
w: `w`(torch.Tensor) is the batch of $w$ of shape `[batch_size, d_latent]`;
x: `x`(torch.Tensor) are the generated images of shape `[batch_size, 3, height, width]`.
It should return:
loss: the penalty

You just need to implement the algorithm module; no need to provide corresponding examples