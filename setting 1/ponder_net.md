You should implement the PonderNet using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 

PonderNet adapts the computation based on the input.
It changes the number of steps to take on a recurrent network based on the input.
PonderNet learns this with end-to-end gradient descent.

PonderNet has a step function of the form

$$\hat{y}_n, h_{n+1}, \lambda_n = s(x, h_n)$$

where $x$ is the input, $h_n$ is the state, $\hat{y}_n$ is the prediction at step $n$,
and $\lambda_n$ is the probability of halting (stopping) at current step.

$s$ can be any neural network (e.g. LSTM, MLP, GRU, Attention layer).

The unconditioned probability of halting at step $n$ is then,

$$p_n = \lambda_n \prod_{j=1}^{n-1} (1 - \lambda_j)$$

That is the probability of not being halted at any of the previous steps and halting at step $n$.

During inference, we halt by sampling based on the halting probability $\lambda_n$
 and get the prediction at the halting layer $\hat{y}_n$ as the final output.

During training, we get the predictions from all the layers and calculate the losses for each of them.
And then take the weighted average of the losses based on the probabilities of getting halted at each layer
$p_n$.

The step function is applied to a maximum number of steps donated by $N$.

The overall loss of PonderNet is

\begin{align}
L &= L_{Rec} + \beta L_{Reg} \\
L_{Rec} &= \sum_{n=1}^N p_n \mathcal{L}(y, \hat{y}_n) \\
L_{Reg} &= \mathop{KL} \Big(p_n \Vert p_G(\lambda_p) \Big)
\end{align}

$\mathcal{L}$ is the normal loss function between target $y$ and prediction $\hat{y}_n$.

$\mathop{KL}$ is the [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).

$p_G$ is the [Geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution) parameterized by
$\lambda_p$. *$\lambda_p$ has nothing to do with $\lambda_n$; we are just sticking to same notation as the paper*.
$$Pr_{p_G(\lambda_p)}(X = k) = (1 - \lambda_p)^k \lambda_p$$

The regularization loss biases the network towards taking $\frac{1}{\lambda_p}$ steps and incentivizes
 non-zero probabilities for all steps; i.e. promotes exploration.

You should implement three modules: ParityPonderGRU, ReconstructionLoss, RegularizationLoss

1. ParityPonderGRU
PonderNet with GRU for Parity Task.
This model is for the [Parity Task](../parity.html) where the input is a vector of `n_elems`. 
Each element of the vector is either `0`, `1` or `-1` and the output is the parity  - a binary value that is true if the number of `1`s is odd and false otherwise. The prediction of the model is the log probability of the parity being $1$.

The module should be named ParityPonderGRU.
The init function needs to include the following parameters:
n_elems: `n_elems` is the number of elements in the input vector;
n_hidden: `n_hidden` is the state vector size of the GRU;
max_steps: `max_steps` is the maximum number of steps $N$.

The module needs to include at least the following functions:
1. forward: forward propagation function. 
  It should include the following parameters:
  x: `x` is the input of shape `[batch_size, n_elems]`.
  It should return:
  p: $p_1 \dots p_N$ in a tensor of shape `[N, batch_size]`;
  y: $\hat{y}_1 \dots \hat{y}_N$ in a tensor of shape `[N, batch_size]` - the log probabilities of the parity being $1$;
  p_m: $p_m$ of shape `[batch_size]`;
  y_m: $\hat{y}_m$ of shape `[batch_size]` where the computation was halted at step $m$

2. ReconstructionLoss
It is reconstruction loss.
$$L_{Rec} = \sum_{n=1}^N p_n \mathcal{L}(y, \hat{y}_n)$$
$\mathcal{L}$ is the normal loss function between target $y$ and prediction $\hat{y}_n$.

The module should be named ReconstructionLoss.
The init function needs to include the following parameters:
loss_func: `loss_func`(nn.Module) is the loss function $\mathcal{L}$. 

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
p: `p`(torch.Tensor) is $p_1 \dots p_N$ in a tensor of shape `[N, batch_size]`;
y_hat: `y_hat`(torch.Tensor) is $\hat{y}_1 \dots \hat{y}_N$ in a tensor of shape `[N, batch_size, ...]`;
y: `y`(torch.Tensor) is the target of shape `[batch_size, ...]`.

It should return:
total_loss: The total $\sum_{n=1}^N p_n \mathcal{L}(y, \hat{y}_n)$.

3. RegularizationLoss
$$L_{Reg} = \mathop{KL} \Big(p_n \Vert p_G(\lambda_p) \Big)$$
$\mathop{KL}$ is the [Kullback–Leibler divergence]. $p_G$ is the [Geometric distribution]. 
$\lambda_p$. *$\lambda_p$ has nothing to do with $\lambda_n$; we are just sticking to same notation as the paper*.
$$Pr_{p_G(\lambda_p)}(X = k) = (1 - \lambda_p)^k \lambda_p$$
The regularization loss biases the network towards taking $\frac{1}{\lambda_p}$ steps and incentivies non-zero probabilities for all steps; i.e. promotes exploration.

The module should be named RegularizationLoss.
The init function needs to include the following parameters:
lambda_p: `lambda_p` is $\lambda_p$ - the success probability of geometric distribution;
max_steps: `max_steps` is the highest $N$; we use this to pre-compute $p_G(\lambda_p)$. The default value is 1000.

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
p: `p`(torch.Tensor) is $p_1 \dots p_N$ in a tensor of shape `[N, batch_size]`.
It should return:
kl_divergence: the kl divergence.

You just need to implement the algorithm module; no need to provide corresponding examples