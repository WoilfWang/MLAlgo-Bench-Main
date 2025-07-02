You should implement Adam Optimizer using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 

*Adam* update is,

\begin{align}
m_t &\leftarrow \beta_1 m_{t-1} + (1 - \beta_1) \cdot g_t \\
v_t &\leftarrow \beta_2 v_{t-1} + (1 - \beta_2) \cdot g_t^2 \\
\hat{m}_t &\leftarrow \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &\leftarrow \frac{v_t}{1-\beta_2^t} \\
\theta_t &\leftarrow \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align}

where $\alpha$, $\beta_1$, $\beta_2$ and $\epsilon$ are scalar hyper parameters.
$m_t$ and $v_t$ are first and second order moments.
$\hat{m}_t$  and $\hat{v}_t$ are biased corrected moments.
$\epsilon$ is used as a fix for division by zero error, but also acts as a form of a hyper-parameter
that acts against variance in gradients.

Effective step taken assuming $\epsilon = 0$ is,
$$\Delta t = \alpha \cdot \frac{\hat{m}_t}{\hat{v}_t}$$
This is bounded by,
$$\vert \Delta t \vert \le \alpha \cdot \frac{1 - \beta_1}{\sqrt{1-\beta_2}}$$
when $1-\beta_1 \gt \sqrt{1-\beta_2}$
and
$$\vert \Delta t\vert  \le \alpha$$
otherwise.
And in most common scenarios, $$ \vert \Delta t \vert \approx \alpha $$.

The module should be named Adam.
Adam should inherit pytorch's Optimizer class. (from torch.optim.optimizer import Optimizer)


You may use class WeightDecay. But you don't need to implement WeightDecay. 
The WeightDecay class implements L2 weight decay for regularizing model parameters by call the function \_\_call\_\_ during training.
The \_\_call\_\_ method is used to apply weight decay during the optimization process. It is invoked during the backward pass of the training loop, where the gradient of the parameters is being computed.
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
defaults: `defaults` is a dictionary of default for group values. The default value is None.


The module needs to include at least the following functions:
1. step: The step function in an optimizer is responsible for performing a single optimization step, i.e., it updates the model parameters based on the computed gradients.
It should include the following parameters:
closure (optional): This is an optional function that, when provided, will be called to calculate the loss and perform the backward() pass. If closure is not None, it is executed within a torch.enable_grad() context to ensure that gradients are calculated. The default value is None.
It should return:
loss (optional): The function returns the loss value calculated inside the closure function, if it was provided. If no closure is used, it returns None.

You just need to implement the algorithm module; no need to provide corresponding examples