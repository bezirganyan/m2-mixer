from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, output_dim=None, dropout=0., **kwargs):
        super().__init__()

        self.module_list = nn.ModuleList()
        self.output_dim = output_dim

        for i in range(num_blocks):
            if i == 0:
                self.module_list.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.module_list.append(nn.Linear(hidden_dim, hidden_dim))
            self.module_list.append(nn.ReLU())
            # self.module_list.append(nn.BatchNorm1d(hidden_dim))
            self.module_list.append(nn.Dropout(dropout))

        if output_dim is not None:
            self.module_list.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for module in self.module_list:
            x = module(x)

        return x
