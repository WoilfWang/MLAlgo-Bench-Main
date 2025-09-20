You should implement Dueling Network Model for Q Values using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 

Intuition behind dueling network architecture is that in most states the action doesn't matter, and in some states the action is significant. Dueling network allows this to be represented very well.

\begin{align}
        Q^\pi(s,a) &= V^\pi(s) + A^\pi(s, a)
        \\
        \mathop{\mathbb{E}}_{a \sim \pi(s)}
         \Big[
          A^\pi(s, a)
         \Big]
        &= 0
    \end{align}

    So we create two networks for $V$ and $A$ and get $Q$ from them.
    $$
        Q(s, a) = V(s) +
        \Big(
            A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a')
        \Big)
    $$
    We share the initial layers of the $V$ and $A$ networks.

The module should be named DuelDQN.

The model starts with a series of 3 convolutional layers that process the input frames (such as images from an environment in reinforcement learning). These convolutional layers extract hierarchical features from the input data. The layers are as follows:
	•	First Convolutional Layer:
	•	Input: 4 channels (for example, 4 frames stacked together in a deep Q-learning agent).
	•	Output: 32 channels.
	•	Kernel size: 8x8.
	•	Stride: 4 (downsampling the image to a smaller spatial dimension).
	•	Activation: ReLU.
	•	Output shape: (batch_size, 32, 20, 20).
	•	Second Convolutional Layer:
	•	Input: 32 channels.
	•	Output: 64 channels.
	•	Kernel size: 4x4.
	•	Stride: 2.
	•	Activation: ReLU.
	•	Output shape: (batch_size, 64, 9, 9).
	•	Third Convolutional Layer:
	•	Input: 64 channels.
	•	Output: 64 channels.
	•	Kernel size: 3x3.
	•	Stride: 1.
	•	Activation: ReLU.
	•	Output shape: (batch_size, 64, 7, 7).
Fully Connected Layer:
After the convolutional layers, the output is flattened into a 1D vector and passed through a fully connected (dense) layer:
	•	Input: Flattened vector of size 7 * 7 * 64 (i.e., 3136 features).
	•	Output: 512 features.
	•	Activation: ReLU.
State and Action Value Heads:
The model then splits into two separate heads:
	•	State Value Head:
	•	This head computes the state value  V(s) .
	•	A fully connected layer outputs a scalar value for the state.
	•	Input: 512 features.
	•	Output: A scalar value  V(s) .
	•	Action Value Head:
	•	This head computes the action value  A(s, a)  for each possible action.
	•	A fully connected layer outputs the value of each action.
	•	Input: 512 features.
	•	Output: 4 action values (corresponding to the 4 possible actions in this example).


The initialization function does not contain any input parameters.


The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
obs: represents the input to the network.
It should return:
q: This represents the Q-values for each possible action in the given state.

You just need to implement the algorithm module; no need to provide corresponding examples