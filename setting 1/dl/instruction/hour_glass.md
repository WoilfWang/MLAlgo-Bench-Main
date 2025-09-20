You should implement Hourglass model using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 

This module introduces a hierarchical transformer architecture to handle long sequences efficiently. The first half of the transformer layers down-sample tokens and the second half up-samples with direct skip connections between layers of the same resolution. This is a little similar to U-Net for vision tasks.
They try different up-sampling and down-sampling techniques and build a model with the best performing up and down-sampling techniques which they call the hourglass model.
Here we have implemented the simplest up-sampling and down-sampling techniques for simplicity. We will consider adding more complex (and better performing) implementations later.
This model recursively adds layers to the middle while shortening the sequence by down-sampling. The shortened sequence processed by another hourglass model is sandwiched between two normal transformer layers. (A transformer layer has a self-attention layer and a position-wise feed-forward layer).

The module should be named HourGlass.

The init function needs to include the following parameters:
n_heads: is the number of heads in multi-head attention layers;
d_model: is the size of the token embeddings;
dropout: is the dropout probability;
d_ff: is the dimensionality of the hidden layer in position-wise feed-forward layers;
shortening_factors:is the list of shortening factors;

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
x: the input tensor. It has the shape `[seq_len, batch_size, d_model]`.
It should return:
result: the output of the Hourglass model.

You just need to implement the algorithm module; no need to provide corresponding examples