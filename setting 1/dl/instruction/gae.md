You should implement Generalized Advantage Estimation using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 

The **Generalized Advantage Estimation (GAE)** method, introduced in the paper *"High-Dimensional Continuous Control Using Generalized Advantage Estimation"* by Schulman et al., is an enhancement of the standard Advantage Estimation used in reinforcement learning. GAE is designed to reduce the variance of policy gradient estimates while maintaining low bias. It achieves this by combining Monte Carlo estimates with Temporal Difference (TD) learning.

### Core Concept:
GAE estimates the advantage function $ A_t $, which represents how much better an action $ a_t $ taken at time $ t $ is compared to the expected value of the return. The goal is to balance the trade-off between variance and bias in the estimate.

### Mathematical Foundation:

1. **Advantage Function**:
   The advantage function $ A_t $ is defined as:
   $$
   A_t = Q(s_t, a_t) - V(s_t)
   $$
   where $ Q(s_t, a_t) $ is the action-value function, and $ V(s_t) $ is the value function of state $ s_t $.

2. **TD Error**:
   Temporal difference (TD) error $ \delta_t $ is given by:
   $$
   \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
   $$
   where $ r_t $ is the reward received at time $ t $, $ \gamma $ is the discount factor, and $ V(s_{t+1}) $ is the value of the next state.

3. **Generalized Advantage Estimation (GAE)**:
   GAE combines multiple TD errors to form a smoother and less noisy estimate of the advantage. It introduces a parameter $ \lambda $ to control the trade-off between bias and variance. The GAE at time $ t $ is defined as:
   $$
   A_t^{\lambda} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
   $$
   where $ \delta_{t+l} $ is the TD error at time step $ t+l $, and $ \lambda $ is a hyperparameter between 0 and 1.

   - When $ \lambda = 0 $, GAE reduces to using a one-step TD error ($ \delta_t $), which has low bias but high variance.
   - When $ \lambda = 1 $, GAE approaches the Monte Carlo return, which has low variance but high bias.

4. **Bias-Variance Trade-off**:
   The value of $ \lambda $ controls the trade-off between bias and variance:
   - **High $ \lambda $**: GAE relies more on the Monte Carlo estimates, which have lower variance but higher bias.
   - **Low $ \lambda $**: GAE relies more on the TD error, which reduces bias but increases variance.


The module should be named GAE.
The init function needs to include the following parameters:
n_workers (int): The number of workers (or parallel environment instances) involved in training;
worker_steps (int): The number of time steps each worker (or environment instance) processes. This represents the number of time steps per worker for each forward pass in the algorithm;
gamma (float): The discount factor, used to calculate the discounted future rewards;
lambda_ (float): The $\lambda$ value used in GAE to balance the bias and variance of the advantage estimate. 

The module needs to include at least the following functions:
1. __call__

### Calculate advantages

\begin{align}
        \hat{A_t^{(1)}} &= r_t + \gamma V(s_{t+1}) - V(s)
        \\
        \hat{A_t^{(2)}} &= r_t + \gamma r_{t+1} +\gamma^2 V(s_{t+2}) - V(s)
        \\
        ...
        \\
        \hat{A_t^{(\infty)}} &= r_t + \gamma r_{t+1} +\gamma^2 r_{t+2} + ... - V(s)
        \end{align}

$\hat{A_t^{(1)}}$ is high bias, low variance, whilst
$\hat{A_t^{(\infty)}}$ is unbiased, high variance.

   We take a weighted average of $\hat{A_t^{(k)}}$ to balance bias and variance.
        This is called Generalized Advantage Estimation.
        $$\hat{A_t} = \hat{A_t^{GAE}} = \frac{\sum_k w_k \hat{A_t^{(k)}}}{\sum_k w_k}$$
        We set $w_k = \lambda^{k-1}$, this gives clean calculation for
        $\hat{A_t}$

  \begin{align}
        \delta_t &= r_t + \gamma V(s_{t+1}) - V(s_t)
        \\
        \hat{A_t} &= \delta_t + \gamma \lambda \delta_{t+1} + ... +
                             (\gamma \lambda)^{T - t + 1} \delta_{T - 1}
        \\
        &= \delta_t + \gamma \lambda \hat{A_{t+1}}
        \end{align}

It should include the following parameters:
done (np.ndarray): A 2D array that indicates whether the episode has completed at each time step (or if it’s in a terminal state). The shape is (n_workers, worker_steps). Each element done[i, t] is 1 if worker i has completed the episode at time step t, and 0 if not; 
rewards (np.ndarray): A 2D array representing the reward values at each time step. The shape is (n_workers, worker_steps), and rewards[i, t] is the reward received by worker i at time step t;
values (np.ndarray): 

It should return: 
advantages: A 2D array representing the advantage estimates at each time step. The shape is (n_workers, worker_steps), where each element corresponds to the calculated advantage for the respective worker at that time step.

You just need to implement the algorithm module; no need to provide corresponding examples