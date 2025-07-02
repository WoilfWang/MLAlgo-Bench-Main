You should implement Graph Attention Networks v2 (GATv2) using python and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


Graph Attention Networks v2 (GATv2) is a neural network architecture that operates on graph-structured data similar to Graph Attention Network (GAT). A graph consists of nodes and edges connecting nodes. For example, in Cora dataset the nodes are research papers and the edges are citations that connect the papers.

Static attention is when the attention to the key nodes has the same rank (order) for any query node. GATs static attention mechanism fails on some graph problems with a synthetic dictionary lookup dataset. It's a fully connected bipartite graph where one set of nodes (query nodes) have a key associated with it and the other set of nodes have both a key and a value associated with it. The goal is to predict the values of query nodes. GAT fails on this task because of its limited static attention.

GAT computes the attention from query node $i$ to key node $j$ as:

$\begin{align}
e_{ij} &= \text{LeakyReLU} \Big(\mathbf{a}^\top \Big[
 \mathbf{W} \overrightarrow{h_i} \Vert  \mathbf{W} \overrightarrow{h_j}
\Big] \Big) \\
&=
\text{LeakyReLU} \Big(\mathbf{a}_1^\top  \mathbf{W} \overrightarrow{h_i} +
 \mathbf{a}_2^\top  \mathbf{W} \overrightarrow{h_j}
\Big)
\end{align}$

Note that for any query node $i$, the attention rank ($argsort$) of keys depends only on $\mathbf{a}_2^\top  \mathbf{W} \overrightarrow{h_j}$. Therefore the attention rank of keys remains the same (static) for all queries.

The GATv2 operator fixes the static attention problem of the standard [GAT](../gat/index.html).  GATv2 allows dynamic attention by changing the attention mechanism. It computes the attention from query node $i$ to key node $j$ as the follow formula:

$\begin{align}
e_{ij} &= \mathbf{a}^\top \text{LeakyReLU} \Big( \mathbf{W} \Big[
 \overrightarrow{h_i} \Vert  \overrightarrow{h_j}
\Big] \Big) \\
&= \mathbf{a}^\top \text{LeakyReLU} \Big(
\mathbf{W}_l \overrightarrow{h_i} +  \mathbf{W}_r \overrightarrow{h_j}
\Big)
\end{align}$

The module should be named GraphAttentionV2Layer.
The init function needs to include the following parameters:

in_features: the number of input features per node;
out_features: the number of output features per node;
n_heads: the number of attention heads;
is_concat: whether the multi-head results should be concatenated or averaged. The default value is True;
dropout: the dropout probability. The default value is 0.6;
leaky_relu_negative_slope: the negative slope for leaky relu activation. The default value is 0.2;
share_weights: The default value is False. If set to True, the same matrix will be applied to the source and the target node of every edge.

The module needs to include at least the following functions:

1. forward: forward propagation function. It should include the following parameters:
   h: the tensors that store the input mode embeddings. It has the shape [n_nodes, in_features] .
   adj_mat: the tensors that store the adjacency matrix. Adjacency matrix represent the edges (or connections) among nodes. `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j` . Here it has the shape [n_nodes, n_nodes, 1] since the adjacency matrix for each head is the same.

The return of forward function includes:
attn_res: the results of the graph attention v2 layer. If is_concat=True, the shape of result is [n_nodes, n_heads * n_hidden].

You just need to implement the algorithm module; no need to provide corresponding examples.