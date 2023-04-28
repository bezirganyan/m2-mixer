import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRU(nn.Module):
    """
    Implements Gated Recurrent Unit (GRU).
    Taken from MultiBench implementation: https://github.com/pliang279/MultiBench
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1, flatten=False, has_padding=False, last_only=False,
                 batch_first=True, **kwargs):
        """Initialize GRU Module.

        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            dropout (bool, optional): Whether to apply dropout layer or not. Defaults to False.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            flatten (bool, optional): Whether to flatten output before returning. Defaults to False.
            has_padding (bool, optional): Whether the input has padding or not. Defaults to False.
            last_only (bool, optional): Whether to return only the last output of the GRU. Defaults to False.
            batch_first (bool, optional): Whether to batch before applying or not. Defaults to True.
        """
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.flatten = flatten
        self.has_padding = has_padding
        self.last_only = last_only
        self.batch_first = batch_first

    def forward(self, x):
        """Apply GRU to input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if self.has_padding:
            x = pack_padded_sequence(
                x[0], x[1], batch_first=self.batch_first, enforce_sorted=False)
            out = self.gru(x)[1][-1]
        elif self.last_only:
            out = self.gru(x)[1][0]

            return out
        else:
            out, l = self.gru(x)
        if self.dropout:
            out = self.dropout_layer(out)
        if self.flatten:
            out = torch.flatten(out, 1)

        return out

