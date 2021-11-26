import torch
from torch import nn
import torch.nn.functional as F


class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size, use_bias=True):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_shape: The shape of the input data, in the form (b, f)
        :param num_filters: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(LinearLayer, self).__init__()

        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.ones(output_size, input_size))
        nn.init.xavier_uniform_(self.weight)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        weight, bias, = (
            None,
            None,
        )

        if weight is None:
            if self.use_bias:
                weight, bias = self.weight, self.bias
            else:
                weight = self.weight
                bias = None

        out = F.linear(input=x, weight=weight, bias=bias)
        return out


class MLPClassifier(nn.Module):
    def __init__(self, input_size, output_size, args):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        # self.linear1 = nn.Linear(input_size, 256)
        # self.linear2 = nn.Linear(256, 64)
        # self.linear3 = nn.Linear(64, output_size)
        self.dense = LinearLayer(input_size, input_size)
        self.out_proj = LinearLayer(input_size, output_size)
        self.dropout = args.dropout
        self.activation = nn.ReLU()

    def forward(self, hidden_states, return_pooled=False):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.out_proj(pooled_output)
        return logits