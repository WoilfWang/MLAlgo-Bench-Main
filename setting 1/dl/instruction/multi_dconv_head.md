You should implement Squared ReLU and Multi-DConv-Head Attention using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 


### Squared ReLU

The most effective modification found by the search is using a square ReLU instead of ReLU in the [position-wise feedforward module](../feed_forward.html).
$$y = {\max(x, 0)}^2$$

The Squared ReLU should be named SquaredReLU.
The init function shouldn't include any parameters.
The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
x: the input tensor.
It should return:
x: the output of the Squared ReLU.

### Multi-DConv-Head Attention (MDHA)

The next effective modification is a depth-wise $3 \times 1$ convolution after multi-head projection for queries, keys, and values.
The convolution is along the sequence dimension and per channel (depth-wise).
To be clear, if the number of channels in each head is $d_k$ the convolution will have $1 \times 3$ kernels for each of the $d_k$ channels.

The module should be named MultiDConvHeadAttention.
The init function needs to include the following parameters:
heads: the nums of the attention head;
d_model: d_model is the number of features in the query , key and value vectors;
dropout_prob: the proportion of neuron dropout. The default value is 0.1.
The model needs to include at least the following functions:
1. forward: forward propagation function. It should include the following parameters:
   query: the tensors that store collection of query. It has the shape `[seq_len, batch_size, d_model]`;
   value: the tensors that store collection of value. It has the shape `[seq_len, batch_size, d_model]`;
   key: the tensors that store collection of key. It has the shape `[seq_len, batch_size, d_model]`;
   mask: `mask` has shape `[seq_len, seq_len, batch_size]` and `mask[i, j, b]` indicates whether for batch `b` , query at position `i` has access to key-value at position `j`.
   The return of forward function includes:
   x: the results of the Multi-DConv-Head Attention

You just need to implement the algorithm module; no need to provide corresponding examples.