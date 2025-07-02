You should implement DeepNorm using python and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


DeepNorm is a method to stabilize extremely deep transformers through a new normalizing function to replace LayerNorm and a weight initialization scheme. This combines the performance of Post-LayerNorm and the stability of Pre-LayerNorm. Transformers with DeepNorms are supposed to be stable even without a learning rate warm-up.
It is shown that the changes to layer outputs (for the same input) change gradually during stable training; when unstable it changes rapidly during the initial training steps. This happens with initializing weights to small values, and learning rate warm-ups where the training is stable. DeepNorm uses the idea of keeping the changes to layer outputs small to derive the new normalization and weight initialization mechanism.
Usually, the weights are initialized with Xavier or Kaiming initializations. DeepNorm suggests scaling the weights of the two linear transforms in the Feed-Forward Network, the value projection transform, and the output projection transform of the attention layer. Weights of these transforms are scaled by (has a gain equal to) $\beta$.
The scaling is implemented in the Normalization Function:

$$
x_{l + 1} = \mathop{LN}\Big( \alpha x_l + \mathop{G_l} \big(x_l, \theta_l \big)\Big)
$$
where $\alpha$ is a constant that depends on the depth of the transformer, $\mathop{LN}$ is Layer Normalization, and $\mathop{G}_l (x_l, \theta_l)$ is the function of the $l$-th transformer sub-layer (FFN or attention). This function is used to replace Post-LayerNorm.

The value of $\alpha$ and $\beta$ constants:
$$
\begin{align}
\begin{array} {c|cc|cc}
\text{Type} & \text{Enc-} \alpha & \text{Enc-} \beta &  \text{Dec-} \alpha & \text{Dec-} \beta \\
\hline \\
\text{Encoder only} & (2N)^{\frac{1}{4}} & (8N)^{-\frac{1}{4}} & - & - \\
\text{Decoder only} & - & - & (2M)^{\frac{1}{4}} & (8M)^{-\frac{1}{4}} \\
\text{Enc-Dec} & 0.81 (N^4M)^{\frac{1}{16}} & 0.87 (N^4 M)^{-\frac{1}{16}} &
 (3M)^{\frac{1}{4}} & (12M)^{-\frac{1}{4}} \\
\end{array}
\end{align}
$$
Where $N$ is the number of layers in the encoder and $M$​ is the number of layers in the decoder.
The module should be named DeepNormTransformerLayer.

The DeepNormTransformerLayer module implements a transformer decoder layer with DeepNorm.
The init function needs to include the following parameters:
d_model: the token embedding size;
self_attn: the self attention module. The module is named MultiHeadAttention. You can code with "from labml_nn.transformers import MultiHeadAttention" to import it. 
feed_forward: the feed forward module. The module is named FeedForward. You can code with "from labml_nn.transformers.feed_forward import FeedForward" to import it. 
deep_norm_alpha: $\alpha$ coefficient in DeepNorm;
deep_norm_beta: $\beta$​ constant for scaling weights initialization;
The module should contain the following function:
1. forward.
It should include the following parameter:
x: x(torch.Tensor) are the embeddings of shape [seq_len, batch_size, d_model].
It return:
retult: the result of deep norm transformer layer.

Here is the brief description of some modules that you may use.

class MultiHeadAttention(nn.Module)
This computes scaled multi-headed attention for given `query`, `key` and `value` vectors as: $\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$. In simple terms, it finds keys that matches the query, and gets the values of those keys. It uses dot-product of query and key as the indicator of how matching they are. Before taking the softmax, the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$. Softmax is calculated along the axis of of the sequence (or time).
The forward function includes the following input parameters:
query: the tensors that store collection of query. It has the shape [seq_len, batch_size, d_model].
value: the tensors that store collection of value. It has the shape [seq_len, batch_size, d_model].
key: the tensors that store collection of key. It has the shape [seq_len, batch_size, d_model].
mask: mask has shape [seq_len, seq_len, batch_size] and mask[i, j, b] indicates whether for batch b , query at position i has access to key-value at position j.
The return of forward function includes:
x: the results of the multi-head attention.

class FeedForward(Module)
FFN consists of two fully connected layers. Number of dimensions in the hidden layer $d_{ff}$, is generally set to around four times that of the token embedding $d_{model}$. So it is sometime also called the expand-and-contract network. The FFN function is, $FFN(x, W_1, W_2, b_1, b_2) = \max(0, x W_1 + b_1) W_2 + b_2$ where $W_1$, $W_2$, $b_1$ and $b_2$​ are learnable parameters.
The forward function includes the following input parameters:
x: the tensors that store the input of network.
The return of forward function includes:
ffn_x: the results of the feed-forward network. 
The DeepNormTransformerLayer module needs to include at least the following functions:

1. forward: forward propagation function. It should include the following parameters:
   ​x: the tensors that store the input of transformer layer.
   The return of forward function includes:
   y: the output of DeepNorm layer.
You just need to implement the algorithm module; no need to provide corresponding examples.