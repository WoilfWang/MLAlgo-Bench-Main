You should implement Mult-Head Attention with Linear Biases using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 

This replaces positional encodings with biases added to attention scores (attention logits, before the softmax).
This is a relative scheme tested on autoregressive tasks, and the bias is higher for closeby tokens and lower for far-away tokens.
The biases decrease linearly in the log scale (because it's before the softmax) and each head has a different slope.

Here's the attention formula for $i$-th token,
\begin{align}
\mathbf{a}_i
&= \text{softmax} \bigg( \mathbf{q}_i \mathbf{K}^\top + m \cdot \big[-(i-1), \dots, -1, 0 \big] \bigg) \\
&= \text{softmax} \bigg( \mathbf{q}_i \mathbf{K}^\top + m \cdot \big[0, 1, \dots, (i - 1) \big] \bigg)
\end{align}

where $\mathbf{q}_i \in \mathbb{R}^d$ is the query of the $i$-th token, $K \in \mathbb{R}^{i \times d}$ are the keys up to $i$, and $d$ the number of features per head. Note that the above equality halts because $\text{softmax}$ is invariant to translations (you can add any constant to all elements without changing the result).
The module should be named AlibiMultiHeadAttention.

The init function needs to include the following parameters:
heads: the nums of the attention head;
d_model: d_model is the number of features in the query , key and value vectors;
dropout_prob: the proportion of neuron dropout. The default value is 0.1;

The model needs to include at least the following functions:
1. forward: forward propagation function. It should include the following parameters:
   query: the tensors that store collection of query. It has the shape `[seq_len, batch_size, d_model]`;
   value: the tensors that store collection of value. It has the shape `[seq_len, batch_size, d_model]`;
   key: the tensors that store collection of key. It has the shape `[seq_len, batch_size, d_model]`;
   mask: `mask` has shape `[seq_len, seq_len, batch_size]` and `mask[i, j, b]` indicates whether for batch `b` , query at position `i` has access to key-value at position `j`.
   The return of forward function includes:
   x: the results of the multi-head attention


You just need to implement the algorithm module; no need to provide corresponding examples