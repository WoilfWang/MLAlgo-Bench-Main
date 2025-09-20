You should implement Sophia Optimizer using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 

Sophia is more adaptive to heterogeneous curvatures than Adam, more resistant
to non-convexity and rapid change of Hessian than Newton’s method, and also uses a low-cost
pre-conditioner.

Sophia keeps diagonal Hessian estimates with EMA across iterations.
The diagonal Hessian $\hat{h}_t$ is calculated every $k$ steps.

\begin{align}
h_t = \beta_2 h_{t-k} + (1 - \beta_2) \hat{h}_t \ \ \ \ \text{ if } t \text{ mod } k = 1; \text{ else }  h_t = h_{t-1}
\end{align}

Sophia uses EMA of gradients $m_t$, only considers positive entries of
 the diagonal Hessian and does per-coordinate clipping to the update.

\begin{align}
m_t &\leftarrow \beta_1 m_{t-1} + (1 - \beta_1)g_t \\
\theta_{t + 1} &\leftarrow \theta_t - \eta \cdot \operatorname{clip} \bigg(\frac{m_t}{ \max \{h_t, \epsilon \} }, \rho \bigg)
\end{align}

where $\epsilon$ is a very small value to prevent division by $0$.

### Gauss-Newton-Bartlett (GNB) estimator

\begin{align}
\hat{L}(\theta) &= \frac{1}{B} \sum^{B}_{b=1} \ell_{CE} \big( f(\theta, x_b), \hat{y}_b \big) \\
\hat{h}_t &= B \cdot \nabla_\theta \hat{L} (\theta) \odot \nabla_\theta \hat{L} (\theta)
\end{align}

where $x_b$ are the inputs,
$B$ is the batch size (number of inputs/tokens),
$\ell_{CE}$ is cross entropy loss, and
$\hat{y}_b$ are sampled from the logits $f(\theta, x_b)$.

Note that this hessian estimate is always positive and therefore we
can replace $\max \{h_t, \epsilon \}$ with $h_t + \epsilon$.

Sophia with Gauss-Newton-Bartlett (GNB) estimator is **Sophia-G**

The module should be named Sophia.
Sophia should inherit pytorch's Optimizer class. (from torch.optim.optimizer import Optimizer).

You may use class WeightDecay. But you don't need to implement WeightDecay. 
The WeightDecay class implements L2 weight decay for regularizing model parameters by call the function __call__ during training.
The __call__ method is used to apply weight decay during the optimization process. It is invoked during the backward pass of the training loop, where the gradient of the parameters is being computed.
The input parameters contain:
param (torch.nn.Parameter): This is the model parameter (weight) to which the weight decay will be applied;
grad (torch.Tensor): This is the gradient of the parameter (param), which is computed by backpropagation;
group (Dict[str, any]): A dictionary containing the hyperparameters for the current optimization group.
The return of __call__:
The output is the possibly modified gradient. If self.weight_decouple is True, the gradient remains unchanged and is returned as-is. If self.weight_decouple is False, the weight decay is added to the gradient, and the modified gradient is returned.


The init function needs to include the following parameters:
params: `params` is the list of parameters;
lr: `lr` is the maximum learning rate $\eta \rho$. The default value is 1e-4;
betas: `betas`(Tuple[float, float]) is a tuple of ($\beta_1$, $\beta_2$). The default values are (0.9, 0.95);
eps: `eps` is $\epsilon$. The default value is 1e-12;
rho: `rho` is $\rho$. The default value is 0.03;
weight_decay: `weight_decay` is an instance of class `WeightDecay;
defaults: `defaults`(Optional[Dict[str, Any]]) is a dictionary of default for group values. The default value is None.

The module needs to include at least the following functions:
1. step: The step function in an optimizer is responsible for performing a single optimization step, i.e., it updates the model parameters based on the computed gradients.
It should include the following parameters:
closure (optional): This is an optional function that, when provided, will be called to calculate the loss and perform the backward() pass. If closure is not None, it is executed within a torch.enable_grad() context to ensure that gradients are calculated. The default value is None.
It should return:
loss (optional): The function returns the loss value calculated inside the closure function, if it was provided. If no closure is used, it returns None.

You just need to implement the algorithm module; no need to provide corresponding examples