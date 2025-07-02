You should implement MLP-Mixer using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 

This paper applies the model on vision tasks. The model is similar to a transformer with attention layer being replaced by a MLP that is applied across the patches (or tokens in case of a NLP task). 

The module should be named MLPMixer.
This module is a drop-in replacement for self-attention layer. It transposes the input tensor before feeding it to the MLP and transposes back, so that the MLP is applied across the sequence dimension (across tokens or image patches) instead of the feature dimension.

The init function needs to include the following parameters:
mlp: mlp is the ffn module.

The module needs to include at least the following functions:
1. forward: forward propagation function. 
The [normal attention module](../mha.html) can be fed with different token embeddings for $\text{query}$,$\text{key}$, and $\text{value}$ and a mask. We follow the same function signature so that we can replace it directly. For MLP mixing, $$x = \text{query} = \text{key} = \text{value}$$ and masking is not possible.
Shape of `query` (and `key` and `value`) is `[seq_len, batch_size, d_model]`.
It should include the following parameters:
query: the tensors that store collection of query. It has the shape `[seq_len, batch_size, d_model]`;
value: the tensors that store collection of value. It has the shape `[seq_len, batch_size, d_model]`;
key: the tensors that store collection of key. It has the shape `[seq_len, batch_size, d_model]`;
mask: the mask matrix.

It should return:
x: the results of the MLP-Mixer.

You just need to implement the algorithm module; no need to provide corresponding examples