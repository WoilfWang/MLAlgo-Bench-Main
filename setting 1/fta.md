You should implement Fuzzy Tiling Activations using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 

Fuzzy tiling activations are a form of sparse activations based on binning.

Binning is classification of a scalar value into a bin based on intervals.
One problem with binning is that it gives zero gradients for most values (except at the boundary of bins).
The other is that binning loses precision if the bin intervals are large.

FTA overcomes these disadvantages.
Instead of hard boundaries like in Tiling Activations, FTA uses soft boundaries
between bins.
This gives non-zero gradients for all or a wide range of values.
And also doesn't lose precision since it's captured in partial values.

#### Tiling Activations

$\mathbf{c}$ is the tiling vector,

$$\mathbf{c} = (l, l + \delta, l + 2 \delta, \dots, u - 2 \delta, u - \delta)$$

where $[l, u]$ is the input range, $\delta$ is the bin size, and $u - l$ is divisible by $\delta$.

Tiling activation is,

$$\phi(z) = 1 - I_+ \big( \max(\mathbf{c} - z, 0) + \max(z - \delta - \mathbf{c}) \big)$$

where $I_+(\cdot)$ is the indicator function which gives $1$ if the input is positive and $0$ otherwise.

Note that tiling activation gives zero gradients because it has hard boundaries.

#### Fuzzy Tiling Activations

The fuzzy indicator function,

$$I_{\eta,+}(x) = I_+(\eta - x) x + I_+ (x - \eta)$$

which increases linearly from $0$ to $1$ when $0 \le x \lt \eta$
and is equal to $1$ for $\eta \le x$.
$\eta$ is a hyper-parameter.

FTA uses this to create soft boundaries between bins.

$$\phi_\eta(z) = 1 - I_{\eta,+} \big( \max(\mathbf{c} - z, 0) + \max(z - \delta - \mathbf{c}, 0) \big)$$

[Here's a simple experiment](experiment.html) that uses FTA in a transformer.

The module should be named FTA. 
The init function needs to include the following parameters:
lower_limit: is the lower limit $l$;
upper_limit: is the upper limit $u$;
delta: is the bin size $\delta$;
eta: is the parameter $\eta$ that detemines the softness of the boundaries.

It contains the class variable expansion_factor, which is an int variable. The input vector expands by a factor equal to the number of bins ￼.

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
z: the input tensor. 
It should return:
result: the result after Fuzzy Tiling Activation.

You just need to implement the algorithm module; no need to provide corresponding examples