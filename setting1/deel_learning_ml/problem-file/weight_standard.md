You should implement Weight Standardization using python and pytorch from scratch.


Batch normalization gives a smooth loss landscape and avoids elimination singularities. Elimination singularities are nodes of the network that become useless (e.g. a ReLU that gives 0 all the time).
However, batch normalization doesn't work well when the batch size is too small, which happens when training large networks because of device memory limitations. Weight Standardization with Batch-Channel Normalization is a better alternative.
Weight Standardization: 
1. Normalizes the gradients 
2. Smoothes the landscape (reduced Lipschitz constant) 
   The Lipschitz constant is the maximum slope a function has between two points. That is, $L$ is the Lipschitz constant where $L$ is the smallest value that satisfies, $\forall a,b \in A: \lVert f(a) - f(b) \rVert \le L \lVert a - b \rVert$ where $f: A \rightarrow \mathbb{R}^m, A \in \mathbb{R}^n$.
3. Avoids elimination singularities
   Elimination singularities are avoided because it keeps the statistics of the outputs similar to the inputs. So as long as the inputs are normally distributed the outputs remain close to normal. This avoids outputs of nodes from always falling beyond the active range of the activation function (e.g. always negative input for a ReLU).

You just need to implement a function. The function should be named weight_standardization.

Weight standardization computes as:

$$
\hat{W}_{i,j} = \frac{W_{i,j} - \mu_{W_{i,\cdot}}} {\sigma_{W_{i,\cdot}}}
$$
where,
$$
\begin{align}
W &\in \mathbb{R}^{O \times I} \\
\mu_{W_{i,\cdot}} &= \frac{1}{I} \sum_{j=1}^I W_{i,j} \\
\sigma_{W_{i,\cdot}} &= \sqrt{\frac{1}{I} \sum_{j=1}^I W^2_{i,j} - \mu^2_{W_{i,\cdot}} + \epsilon} \\
\end{align}
$$
For a 2D-convolution layer, $O$ is the number of output channels ($O = C_{out}$) and $I$ is the number of input channels times the kernel size ($I = C_{in} \times k_H \times k_W$).
The weight_standardization function should include the following parameters:

        weight: the tensors that store the input weight. It has the shape [out_channels, in_channels].
        eps: hyperparameter $\epsilon$.
The return of function includes:

        result_weight: the results of weight standardization. It has the same shape with input weight.
You just need to implement the algorithm function; no need to provide corresponding examples, and no need to output any other content.