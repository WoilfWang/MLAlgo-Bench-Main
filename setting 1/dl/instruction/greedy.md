You should implement greedy sampling using python and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


Greedy sampling is to sample the most likely token from the distribution of logits.
The module should be named GreedySampler.
This module should inherit from Sampler module. You can code with "from labml_nn.sampling import Sampler" to import that module.
The module needs to include at least the following functions:
1. `__call__`: function to sample. It should include the following parameters:
   logits: the tensors that store the distribution of logits.
   The return of `__call__` function includes:
   res: the results of greedy sampling.
You just need to implement the algorithm module; no need to provide corresponding examples.