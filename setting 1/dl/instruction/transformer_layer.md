You should implement the transformer layer using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


The module should be named TransformerLayer.

The init function needs to include the following parameters:
d_model: d_model is the token embedding size;
d_ff: d_ff is the number of features in the hidden layer of the FFN;
heads: heads is the number of features in the `query`, `key` and `value` vectors.
drouput_prob: dropout_prob is the probability of dropping out after self attention and FFN.

It has a class variable: size = d_model.

The model needs to include at least the following functions:
1. forward. forward propagation function. It should include the following parameters:
x: the input tensor. It has shape [seq_len, batch_size, d_model];
mask: mask has shape [seq_len, seq_len, batch_size] and mask[i, j, b] indicates whether for batch b , query at position i has access to key-value at position j
The return of forward should include:
result: The result of the tranformer layer.

You just need to implement the algorithm module; no need to provide corresponding examples