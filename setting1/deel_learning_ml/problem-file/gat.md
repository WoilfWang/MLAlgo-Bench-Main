You should implement graph attention networks (GAT) using python and pytorch from scratch.

A Graph Attention Network is a neural network architecture that operates on graph-structured data. A graph consists of nodes and edges connecting nodes. For example, in Cora dataset the nodes are research papers and the edges are citations that connect the papers.
GAT leverages masked self-attention to address the shortcomings of prior methods based on graph convolutions or their approximations. GAT consists of graph attention layers stacked on top of each other. Each graph attention layer gets node embeddings as inputs and outputs transformed embeddings. The node embeddings pay attention to the embeddings of other nodes it's connected to. The details of graph attention layers are included alongside the implementation.

The module should be named GraphAttentionLayer.

This is a single graph attention layer. A GAT is made up of multiple such layers.
It takes
$$\mathbf{h} = \{ \overrightarrow{h_1}, \overrightarrow{h_2}, \dots, \overrightarrow{h_N} \}$$
, where $\overrightarrow{h_i} \in \mathbb{R}^F$ as input and outputs,  $$\mathbf{h'} = \{ \overrightarrow{h'_1}, \overrightarrow{h'_2}, \dots, \overrightarrow{h'_N} \}$$, where $\overrightarrow{h'_i} \in \mathbb{R}^{F'}$.

The init function needs to include the following parameters:

    in_features: the number of input features per node;
    out_features: the number of output features per node;
    n_heads: the number of attention heads;
    is_concat: whether the multi-head results should be concatenated or averaged. The default value is True;
    dropout: the dropout probability. The default value is 0.6;
    leaky_relu_negative_slope: the negative slope for leaky relu activation. The default value is 0.2.
The module needs to include at least the following functions:

forward: forward propagation function. It should include the following parameters:

    h: the tensors that store the input mode embeddings. It has the shape [n_nodes, in_features] .
    adj_mat: the tensors that store the adjacency matrix. Adjacency matrix represent the edges (or connections) among nodes. `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j` . Here it has the shape [n_nodes, n_nodes, 1] since the adjacency matrix for each head is the same.
    
The return of forward function includes:

    attn_res: the results of the graph attention layer. If is_concat=True, the shape of result is [n_nodes, n_heads * n_hidden].

You just need to implement the algorithm module; no need to provide corresponding examples, and no need to output any other content.