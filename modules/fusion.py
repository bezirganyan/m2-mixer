import torch
from torch import nn


class BiModalGatedUnit(nn.Module):
    def __init__(self, mod1_in, mod2_in, out_size, **kwargs):
        super(BiModalGatedUnit, self).__init__()
        self.out_size = out_size
        self.mod1_hidden = nn.Linear(mod1_in, out_size)
        self.mod2_hidden = nn.Linear(mod2_in, out_size)

        self.z_hidden = nn.Linear(mod1_in + mod2_in, out_size)

    def forward(self, mod1, mod2):
        mod1_hidden = torch.tanh(self.mod1_hidden(mod1))
        mod2_hidden = torch.tanh(self.mod2_hidden(mod2))

        z_hidden = self.z_hidden(torch.cat([mod1, mod2], dim=-1))
        z = torch.sigmoid(z_hidden)

        return z * mod1_hidden + (1 - z) * mod2_hidden

    def get_output_shape(self, *args, dim=None):
        """
        Returns the output shape of the layer given the input shape.

        Parameters
        ----------
        *args : tuple, list, torch.Size, int
            The input shape of the layer. If a tuple, list, or torch.Size, then the full shape is expected. If an int,
            then the dimension parameter is also expected, and the result will be the output shape of that dimension.
        dim : int, optional
            The dimension of the input shape. Only used if the first argument is an int. Defaults to None. If not None,
            then the args argument is expected to be an int and match the input shape of at the given dimension.
            then the result will be the output shape of that dimension. Since this fusion performs the transformation
            on the last dimension, to get the shape of the last dimension, set dim to -1. Otherwise, the output value
            will be equal to the input value.

        Returns
        -------
        tuple, int

        """
        if dim is not None:
            if not isinstance(args[0], int):
                raise ValueError("The dim argument is only used if the first argument is an int.")
            if dim == -1:
                return self.out_size
            else:
                return args[0][dim] if isinstance(args[0], list | tuple | torch.Size) else args[0]
        shape1 = list(args[0])
        shape1[-1] = self.out_size
        return tuple(shape1)


class ConcatFusion:
    def __init__(self, dim=1, **kwargs):
        self.dim = dim

    def __call__(self, *args):
        return torch.cat(args, dim=self.dim)

    def get_output_shape(self, *args, dim=None):
        """
        Returns the output shape of the layer given the input shape.
        Parameters
        ----------
        *args : tuple, list, torch.Size, int
            The input shape of the layer. If a tuple, list, or torch.Size, then the full shape is expected. If an int,
            then the dimension parameter is also expected, and the result will be the output shape of that dimension.
        dim : int, optional
            The dimension of the input shape. Only used if the first argument is an int. Defaults to None. If not None,
            then the args argument is expected to be an int and match the input shape of at the given dimension.

        Returns
        -------
        tuple, int

        """
        if dim is not None:
            if not isinstance(args[0], int):
                raise ValueError("The dim argument is only used if the first argument is an int.")
            if dim == self.dim:
                return args[0] * 2
            else:
                return args[0]
        shape = list(args[0])
        for arg in args[1:]:
            shape[self.dim] += arg[self.dim]
        return tuple(shape)


class MaxFusion:
    def __init__(self, **kwargs):
        pass

    def __call__(self, *args):
        return torch.maximum(*args)

    @staticmethod
    def get_output_shape(*args, dim=None):
        if dim is not None:
            if not isinstance(args[0], int):
                raise ValueError("The dim argument is only used if the first argument is an int.")
        if args[0] != args[1]:
            raise ValueError("Input shapes must be equal")
        return args[0]


class SumFusion:
    def __init__(self, **kwargs):
        pass

    def __call__(self, *args):
        return torch.add(*args)

    @staticmethod
    def get_output_shape(*args, dim=None, **kwargs):
        if dim is not None:
            if not isinstance(args[0], int):
                raise ValueError("The dim argument is only used if the first argument is an int.")
        if args[0] != args[1]:
            raise ValueError("Input shapes must be equal")
        return args[0]


class MeanFusion:
    def __init__(self, **kwargs):
        pass

    def __call__(self, *args):
        return torch.mean(torch.stack(args), 0)

    @staticmethod
    def get_output_shape(*args, dim=None, **kwargs):
        if dim is not None:
            if not isinstance(args[0], int):
                raise ValueError("The dim argument is only used if the first argument is an int.")
        if args[0] != args[1]:
            raise ValueError("Input shapes must be equal")
        return args[0]
