You should implement PPO using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 

PPO is a policy gradient method for reinforcement learning. Simple policy gradient methods do a single gradient update per sample (or a set of samples). Doing multiple gradient steps for a single sample causes problems because the policy deviates too much, producing a bad policy. PPO lets us do multiple gradient updates per sample by trying to keep the policy close to the policy that was used to sample data. It does so by clipping gradient flow if the updated policy is not close to the policy used to sample the data.

You should implement two modules: PPO Loss, Clipped Value Function Loss.

1. PPO Loss
Here's how the PPO update rule is derived.

  We want to maximize policy reward
     $$\max_\theta J(\pi_\theta) =
       \mathop{\mathbb{E}}_{\tau \sim \pi_\theta}\Biggl[\sum_{t=0}^\infty \gamma^t r_t \Biggr]$$
     where $r$ is the reward, $\pi$ is the policy, $\tau$ is a trajectory sampled from policy,
     and $\gamma$ is the discount factor between $[0, 1]$.

  \begin{align}
    \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
     \sum_{t=0}^\infty \gamma^t A^{\pi_{OLD}}(s_t, a_t)
    \Biggr] &=
    \\
    \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
      \sum_{t=0}^\infty \gamma^t \Bigl(
       Q^{\pi_{OLD}}(s_t, a_t) - V^{\pi_{OLD}}(s_t)
      \Bigr)
     \Biggr] &=
    \\
    \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
      \sum_{t=0}^\infty \gamma^t \Bigl(
       r_t + V^{\pi_{OLD}}(s_{t+1}) - V^{\pi_{OLD}}(s_t)
      \Bigr)
     \Biggr] &=
    \\
    \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
      \sum_{t=0}^\infty \gamma^t \Bigl(
       r_t
      \Bigr)
     \Biggr]
     - \mathbb{E}_{\tau \sim \pi_\theta}
        \Biggl[V^{\pi_{OLD}}(s_0)\Biggr] &=
    J(\pi_\theta) - J(\pi_{\theta_{OLD}})
    \end{align}

  So,
     $$\max_\theta J(\pi_\theta) =
       \max_\theta \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
          \sum_{t=0}^\infty \gamma^t A^{\pi_{OLD}}(s_t, a_t)
       \Biggr]$$

  Define discounted-future state distribution,
     $$d^\pi(s) = (1 - \gamma) \sum_{t=0}^\infty \gamma^t P(s_t = s | \pi)$$

  Then,

  \begin{align}
    J(\pi_\theta) - J(\pi_{\theta_{OLD}})
    &= \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
     \sum_{t=0}^\infty \gamma^t A^{\pi_{OLD}}(s_t, a_t)
    \Biggr]
    \\
    &= \frac{1}{1 - \gamma}
     \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta} \Bigl[
      A^{\pi_{OLD}}(s, a)
     \Bigr]
    \end{align}

  Importance sampling $a$ from $\pi_{\theta_{OLD}}$,

  \begin{align}
    J(\pi_\theta) - J(\pi_{\theta_{OLD}})
    &= \frac{1}{1 - \gamma}
     \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta} \Bigl[
      A^{\pi_{OLD}}(s, a)
     \Bigr]
    \\
    &= \frac{1}{1 - \gamma}
     \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_{\theta_{OLD}}} \Biggl[
      \frac{\pi_\theta(a|s)}{\pi_{\theta_{OLD}}(a|s)} A^{\pi_{OLD}}(s, a)
     \Biggr]
    \end{align}

  Then we assume $d^\pi_\theta(s)$ and  $d^\pi_{\theta_{OLD}}(s)$ are similar.
    The error we introduce to $J(\pi_\theta) - J(\pi_{\theta_{OLD}})$
     by this assumption is bound by the KL divergence between
     $\pi_\theta$ and $\pi_{\theta_{OLD}}$.
    [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528)
     shows the proof of this. I haven't read it.


  \begin{align}
    J(\pi_\theta) - J(\pi_{\theta_{OLD}})
    &= \frac{1}{1 - \gamma}
     \mathop{\mathbb{E}}_{s \sim d^{\pi_\theta} \atop a \sim \pi_{\theta_{OLD}}} \Biggl[
      \frac{\pi_\theta(a|s)}{\pi_{\theta_{OLD}}(a|s)} A^{\pi_{OLD}}(s, a)
     \Biggr]
    \\
    &\approx \frac{1}{1 - \gamma}
     \mathop{\mathbb{E}}_{\textcolor{orange}{s \sim d^{\pi_{\theta_{OLD}}}}
     \atop a \sim \pi_{\theta_{OLD}}} \Biggl[
      \frac{\pi_\theta(a|s)}{\pi_{\theta_{OLD}}(a|s)} A^{\pi_{OLD}}(s, a)
     \Biggr]
    \\
    &= \frac{1}{1 - \gamma} \mathcal{L}^{CPI}
    \end{align}

The module should be named ClippedPPOLoss.
The initialization function does not contain any parameters.

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
log_pi: log_pi (torch.Tensor) is the log probability of the action taken under the current policy $\pi_\theta$. It represents the policy’s output for the action taken given the state during the current iteration;
sampled_log_pi: sampled_log_pi (torch.Tensor) is the log probability of the action taken under the old policy $\pi_{\theta_{OLD}}$. This value is used to calculate the ratio between the current policy and the old policy, as part of the PPO update.
advantage: advantage (torch.Tensor) is the advantage function, which measures how much better (or worse) the action taken is compared to the baseline value function. It helps adjust the policy gradient by weighing the update based on how beneficial the action was.
clip (float): This is the clipping parameter (denoted as $\epsilon$ in the PPO paper). It controls the range in which the policy ratio can deviate from 1. This helps prevent large updates to the policy, ensuring it doesn’t change too drastically in a single update, which would destabilize the learning process.
It should return:
result: The forward function returns the negative mean of the clipped policy reward, which is the objective function that PPO aims to maximize. 

In the forward function, you need to define a class-internal variable, clip_fraction, which you need to calculate yourself.

2. Clipped Value Function Loss

Similarly we clip the value function update also.

  \begin{align}
    V^{\pi_\theta}_{CLIP}(s_t)
     &= clip\Bigl(V^{\pi_\theta}(s_t) - \hat{V_t}, -\epsilon, +\epsilon\Bigr)
    \\
    \mathcal{L}^{VF}(\theta)
     &= \frac{1}{2} \mathbb{E} \biggl[
      max\Bigl(\bigl(V^{\pi_\theta}(s_t) - R_t\bigr)^2,
          \bigl(V^{\pi_\theta}_{CLIP}(s_t) - R_t\bigr)^2\Bigr)
     \biggr]
    \end{align}

  Clipping makes sure the value function $V_\theta$ doesn't deviate
     significantly from $V_{\theta_{OLD}}$.

The module should be named ClippedValueFunctionLoss.
The initialization function does not contain any parameters.

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
value (torch.Tensor): This is the predicted value of the state under the current policy $V^{\pi_\theta}(s_t)$. It is the output of the value function network, representing the expected return from state s_t according to the current policy; 
sampled_value (torch.Tensor): This is the value of the state under the old policy $V^{\pi_{\theta_{OLD}}}(s_t)$, which was used to sample the trajectories in the past. It serves as a reference to prevent large deviations in the value function during updates;
sampled_return (torch.Tensor): This is the target return (often the discounted sum of rewards), which serves as the ground truth for the value function. It is used to compute the loss between the predicted value and the true return;
clip (float): This is the clipping parameter (denoted as \epsilon in the PPO paper). It controls how much the value function update can deviate from the old value function. This ensures that the value function does not change too much in a single update.

It should return:
result:  the value function loss.

You just need to implement the algorithm module; no need to provide corresponding examples