You should implement the Routing among multiple FFNs in Switch Transformer using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


The Switch Transformer uses different parameters for each token by switching among parameters based on the token. Therefore, only a fraction of parameters are chosen for each token. So you can have more parameters but less computational cost.
The switching happens at the Position-wise Feedforward network (FFN) of each transformer block. Position-wise feedforward network consists of two sequentially fully connected layers. In switch transformer we have multiple FFNs (multiple experts), and we chose which one to use based on a router. The output is a set of probabilities for picking a FFN, and we pick the one with the highest probability and only evaluate that. So essentially the computational cost is the same as having a single FFN. In our implementation this doesn't parallelize well when you have many or large FFNs since it's all happening on a single GPU. In a distributed setup you would have each FFN (each very large) on a different device.
The paper introduces another loss term to balance load among the experts (FFNs) and discusses dropping tokens when routing is not balanced.

The module should be named SwitchFeedForward.

The init function needs to include the following parameters:
capacity_factor: capacity_factor is the capacity of each expert as a factor relative to ideally balanced load;
drop_tokens: drop_tokens specifies whether to drop tokens if more tokens are routed to an expert than the capacity;
is_scale_prob: is_scale_prob specifies whether to multiply the input to the FFN by the routing probability;
n_experts: n_experts is the number of experts. Expert denotes FFN layer.
d_model: d_model is the number of features in a token embedding;
d_ff: d_ff is the number of features in the hidden layer of the FFN;
dropout: dropout is dropout probability in the FFN.

The model needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
x: x is the input to the switching module with shape [seq_len, batch_size, d_model].
The return of the function are:
final_output: the final output;
counts: number of tokens routed to each expert;
sum_prob: sum of probabilities for each expert;
len_droppen_tokens: number of tokens dropped;
route_prob_max: routing probabilities of the selected experts.

You just need to implement the algorithm module; no need to provide corresponding examples