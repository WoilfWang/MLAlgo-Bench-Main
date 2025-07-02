You should implement the Primer module using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 

The authors do an evolutionary search for transformer architectures.
They name the architecture found using the search Primer (PRIMitives searched transformER).
**Primer EZ** is the architecture with the two most robust modifications in Primer compared to the original transformer.
Primer EZ trains a lot faster than the vanilla transformer.


### Multi-Head Attention Module

This computes scaled multi-headed attention for given `query`, `key` and `value` vectors.

$$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$

In simple terms, it finds keys that matches the query, and gets the values of those keys.

It uses dot-product of query and key as the indicator of how matching they are. Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$. This is done to avoid large dot-product values causing softmax to give very small gradients when $d_k$ is large.
Softmax is calculated along the axis of of the sequence (or time).

### Squared ReLU

The most effective modification found by the search is using a square ReLU instead of ReLU in
the [position-wise feedforward module](../feed_forward.html).

$$y = {\max(x, 0)}^2$$

### Multi-DConv-Head Attention (MDHA)

The next effective modification is a depth-wise $3 \times 1$ convolution after multi-head projection
 for queries, keys, and values.
The convolution is along the sequence dimension and per channel (depth-wise).
To be clear, if the number of channels in each head is $d_k$ the convolution will have $1 \times 3$
kernels for each of the $d_k$ channels.

You should implement two modules: Squared ReLU activation, Multi-DConv-Head Attention (MDHA).

1. Squared ReLU activation
Squared ReLU is used as the activation function in the position wise feedforward module.
The module should be named SquaredReLU.

The initialization function does not contain any parameters.

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
x: the input tensor.
It should return:
result: the result of squared reLU activation.


2. Multi-DConv-Head Attention (MDHA)
The module should be named MultiDConvHeadAttention.

The init function should contain the following parameters:
heads: heads is the number of heads;
d_model: d_model is the number of features in the query , key and value vectors;
bias: The default value is True.

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should contain the following parameters:
query: query(torch.Tensor) has shape [seq_len, batch_size, d_model];
key: key(torch.Tensor) has shape [seq_len, batch_size, d_model];
value: value(torch.Tensor) has shape [seq_len, batch_size, d_model];
mask: mask(torch.Tensor) has shape [seq_len, seq_len, batch_size] and mask[i, j, b] indicates whether for batch b , query at position i has access to key-value at position j. The default value is None.
It should return:
result: the result after Multi-DConv-Head Attention.


You just need to implement the algorithm module; no need to provide corresponding examples