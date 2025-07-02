You should implement Rectified Adam (RAdam) optimizer using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 

Adam optimizer sometimes converges to a bad local optima during the initial stages of the training;
especially when training transformers.
Researches use warmups to counter this; for the the initial training steps (warm-up stage)
they use a low learning rate.
This paper identifies the problem to be the high variance of adaptive learning rate
during initial stages of training, and counters it using a new rectification term to
reduce variance.

The paper also evaluates two variance reduction mechanisms:
* **Adam-2k**: Only compute the adaptive learning rate ($v_t$ in [Adam](adam.html)) during the first 2k steps,
without changing parameters or calculating momentum ($m_t$).
* **Adam-eps**: Adam with large $\epsilon \approx 10^{-4}$.

## Rectified Adam

Let $\sigma(g_1, ..., g_t)$ and $\psi(g_1, ..., g_t)$ be the functions to calculate
momentum and adaptive learning rate.
For Adam, they are

\begin{align}
\sigma(g_1, ..., g_t) &=  \frac{(1 - \beta_1)\sum_{i=1}^t \beta_1^{t-i} g_i}{1 - \beta_1^t} \\
\psi(g_1, ..., g_t) &=  \sqrt \frac{1 - \beta_2^t}{(1 - \beta_2)\sum_{i=1}^t \beta_2^{t-i} g_i^2}
\end{align}

### Exponential moving average as simple moving average

The distribution of exponential moving average can be approximated as a simple moving average.

$$
\begin{align}
p\Bigg(\frac{(1-\beta_2) \sum_{i=1}^t \beta_2^{t-i} g_i^2}{1 - \beta_2^t} \Bigg) \approx
p\Bigg(\frac{\sum_{i=1}^{f(t,\beta_2)} g_{t+1-i}^2}{f(t,\beta_2)} \Bigg)
\end{align}
$$

Here we are taking the simple moving average of the last $f(t,\beta_2)$ gradients.
$f(t,\beta_2)$ satisfies the following,

\begin{align}
\frac{(1-\beta_2) \sum_{i=1}^t \beta_2^{t-i} \cdot i}{1 - \beta_2^t} =
\frac{\sum_{i=1}^{f(t,\beta_2)} (t+1-i)}{f(t,\beta_2)}
\end{align}

which gives,
$$f(t,\beta_2) = \frac{2}{1-\beta_2} - 1 - \frac{2 t \beta_2^t}{1 - \beta_2^t}$$

### Scaled inverse chi-squared

From above we have
$$
p\Big( \psi^2(g_1, ..., g_t) \Big) \approx
p\Bigg(\frac{\sum_{i=1}^{f(t,\beta_2)} g_{t+1-i}^2}{f(t,\beta_2)} \Bigg)
$$
where $g_i \sim \mathcal{N}(0, \sigma^2)$.
Note that $sigma$ here is the standard deviation and different from $\sigma(.)$ for momentum.

[Scaled inverse chi-squared](https://en.wikipedia.org/wiki/Scaled_inverse_chi-squared_distribution)
is the distribution of squared inverse of mean of $p$ normal distributions.
$$
p\Bigg(\frac{\sum_{i=1}^{f(t,\beta_2)} g_{t+1-i}^2}{f(t,\beta_2)} \Bigg)
\sim
\text{Scale-inv} \mathcal{X}^2(\rho,\frac{1}{\sigma^2})
$$
where $\rho = f(t,\beta_2)$.

### Rectification

They prove that variance of $\psi(.)$ decreases with $\rho$ when
$\psi^2(.) \sim \text{Scale-inv} \mathcal{X}^2(\rho,\frac{1}{\sigma^2})$.

Therefore the variance is minimized at maximal $\rho$ which is
$\rho_{\infty} = \frac{2}{1-\beta_2} - 1$. Let the minimum variance be $C_{\text{var}}$

In order to ensure that the adaptive learning
rate $\psi(.)$ has consistent variance, we rectify the variance with $r$

\begin{align}
r = \sqrt{\frac{C_{\text{var}}}{Var\big[\psi(.)\big]}}
\end{align}

### Approximating $Var[\psi(.)]$

They estimate $Var[\psi(.)] \approx \frac{Var[\psi^2(.)]}{4 \mathbb{E}[\psi^2(.)}$
based on first order expansion of $\sqrt{\psi^2(.)}$
🤪 I didn't get how it was derived.

From $\text{Scale-inv} \mathcal{X}^2$ distribution we have,

\begin{align}
\mathbb{E}\big[\psi^2(.)\big] &= \frac{\rho / \sigma^2}{\rho-2} \\
Var\big[\psi^2(.)\big] &= \frac{2 \rho / \sigma^4}{(\rho-2)^2 (\rho - 2)}
\end{align}

which gives,
$$
Var[\psi(.)] \approx \frac{\rho}{2(\rho-2)(\rho-4)\sigma^2}
$$

### Rectification term

We have

\begin{align}
r &= \sqrt{\frac{C_{\text{var}}}{Var\big[\psi(.)\big]}} \\
Var[\psi(.)] &\approx \frac{\rho}{2(\rho-2)(\rho-4)\sigma^2}
\end{align}

where $C_{\text{var}}$ is $Var\big[\psi(.)\big]$ for $\rho_\infty$.
Lt $\rho$ and step $t$ be $\rho_t$, and $r_t$ be the rectification term
at step $t$.

\begin{align}
C_{\text{var}} &\approx \frac{\rho_\infty}{2(\rho_\infty-2)(\rho_\infty-4)\sigma^2} \\
Var[\psi(g_1,...,g_t)] &\approx \frac{\rho_t}{2(\rho_t-2)(\rho_t-4)\sigma^2}
\end{align}

This gives,

\begin{align}
r_t &= \sqrt{\frac{(\rho_t-2)(\rho_t-4)\rho_\infty}{(\rho_\infty-2)(\rho_\infty-4)\rho_t}}
\end{align}

The module should be named RAdam.
Adam should inherit pytorch's Optimizer class. (from torch.optim.optimizer import Optimizer)

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
lr: lr is the learning rate $\alpha$. The default value is 1e-3;
betas: `betas`(Tuple[float, float]) is a tuple of ($\beta_1$, $\beta_2$). The default values are (0.9, 0.999);
eps: `eps` is $\hat{\epsilon}$ or $\epsilon$ based on `optimized_update`. The default value is 1e-16;
weight_decay: weight_decay is an instance of class WeightDecay;
optimized_update: `optimized_update` is a flag whether to optimize the bias correction of the second moment by doing it after adding $\epsilon$. The default value is True;
amsgrad: `amsgrad` is a flag indicating whether to use AMSGrad or fallback to plain Adam; The default value is False.
degenerated_to_sgd: `degenerated_to_sgd` whether to use sgd when the rectification term $r_t$ is intractable. The default value is True;
defaults: `defaults` is a dictionary of default for group values. The default value is None.

The module needs to include at least the following functions:
1. step: The step function in an optimizer is responsible for performing a single optimization step, i.e., it updates the model parameters based on the computed gradients.
It should include the following parameters:
closure (optional): This is an optional function that, when provided, will be called to calculate the loss and perform the backward() pass. If closure is not None, it is executed within a torch.enable_grad() context to ensure that gradients are calculated. The default value is None.
It should return:
loss (optional): The function returns the loss value calculated inside the closure function, if it was provided. If no closure is used, it returns None.

You just need to implement the algorithm module; no need to provide corresponding examples