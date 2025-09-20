You should implement the transformer xl layer using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


Transformer has a limited attention span, equal to the length of the sequence trained in parallel. All these positions have a fixed positional encoding. Transformer XL increases this attention span by letting each of the positions pay attention to precalculated past embeddings. For instance if the context length is l, it will keep the embeddings of all layers for previous batch of length l and feed them to current step. If we use fixed-positional encodings these pre-calculated embeddings will have the same positions as the current context. They introduce relative positional encoding, where the positional encodings are introduced at the attention calculation. 

The module should be named TransformerXLLayer.

The init function needs to include the following parameters:
heads: heads is the number of features in the `query`, `key` and `value` vectors;
d_model: d_model is the token embedding size;
d_ff: d_ff is the number of features in the hidden layer of the FFN;
dropout_prob: dropout_prob is the probability of dropping out after self attention and FFN;

It has a class variable: size = d_model.

The model needs to include at least the following functions:
1. forward. It should include the following parameters:
x: x is a tensor of the token level feature vectors of shape [seq_len, batch_size, d_model];
mem: mem is a tensor of the past token level feature vectors of shape [mem_len, batch_size, d_model]. mem can be None;
mask: mask is a matrix of shape [seq_len, mem_len + seq_len, batch_size] or [seq_len, mem_len + seq_len, 1] . mask[i, j] is true if token at i can see token at j.
The return of the function should include:
result: The output of the transformer xl layer.

You just need to implement the algorithm module; no need to provide corresponding examples