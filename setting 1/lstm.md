You should implement Long Short-Term Memory (LSTM) using Python and PyTorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) that can learn and remember over long sequences of data. They were introduced to overcome the limitations of traditional RNNs, particularly the vanishing gradient problem that hampers learning long-range dependencies in sequences.
LSTM networks achieve this by introducing a memory cell that can maintain its state over time, and three gates (input, forget, and output gates) that regulate the flow of information into and out of the cell. The cell state and gates are computed using sigmoid and tanh activation functions, which allow the network to selectively remember or forget information.

LSTM Cell computes $c$, and $h$. $c$ is like the long-term memory,
    and $h$ is like the short term memory.
    We use the input $x$ and $h$ to update the long term memory.
    In the update, some features of $c$ are cleared with a forget gate $f$,
    and some features $i$ are added through a gate $g$.

The new short term memory is the $\tanh$ of the long-term memory
    multiplied by the output gate $o$.

Note that the cell doesn't look at long term memory $c$ when doing the update. It only modifies it.
    Also $c$ never goes through a linear transformation.
    This is what solves vanishing and exploding gradients.

Here's the update rule.

\begin{align}
    c_t &= \sigma(f_t) \odot c_{t-1} + \sigma(i_t) \odot \tanh(g_t) \\
    h_t &= \sigma(o_t) \odot \tanh(c_t)
    \end{align}

$\odot$ stands for element-wise multiplication.

Intermediate values and gates are computed as linear transformations of the hidden
    state and input.

\begin{align}
    i_t &= lin_x^i(x_t) + lin_h^i(h_{t-1}) \\
    f_t &= lin_x^f(x_t) + lin_h^f(h_{t-1}) \\
    g_t &= lin_x^g(x_t) + lin_h^g(h_{t-1}) \\
    o_t &= lin_x^o(x_t) + lin_h^o(h_{t-1})
\end{align}

The model needs to include at least the following classes:

1.Class LSTM:
The module should be named LSTM.
The init function needs to include the following parameters:
input_size (int): Number of input features.
hidden_size (int): Number of features in the hidden state.
n_layers (int): Number of LSTM layers.
The forward function( Processes the input sequence through the LSTM network, returning the output sequence and final states.)needs to include the following parameters:
x (torch.Tensor): Input tensor with shape [n_steps, batch_size, input_size].
state (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): Tuple containing the initial  hidden and cell states, each with shape [batch_size, hidden_size]. Defaults to None.
The return of forward function includes:
out (torch.Tensor): Output tensor with shape [n_steps, batch_size, hidden_size].
state (Tuple[torch.Tensor, torch.Tensor]): Tuple containing the final hidden and cell states,  each with shape [batch_size, hidden_size].

You just need to implement the algorithm module; no need to provide corresponding examples.