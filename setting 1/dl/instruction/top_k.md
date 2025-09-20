You should implement top-k sampling using python and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


Top-k sampling is to first pick the top-k tokens from the distribution of logits, and then sample from them.
The module should be named TopKSampler.
This module should inherit from Sampler module. You can code with "from labml_nn.sampling import Sampler" to import that module.
The init function needs to include the following parameters:
k: the number of tokens to pick;
sampler: the sampler to use for the top-k tokens. It can be any sampler that takes a logits tensor as input and returns a token tensor, e.g. "TemperatureSampler";
The module needs to include at least the following functions:
1. `__call__`: function to sample. It should include the following parameters:
   logits: the tensors that store the distribution of logits.
   The return of `__call__` function includes:
   res: the results of top-k sampling.
You just need to implement the algorithm module; no need to provide corresponding examples.