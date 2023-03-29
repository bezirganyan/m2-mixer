import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceClassificationLayer(nn.Module):
    def __init__(self, hidden_dim: int, proj_dim: int, num_classes: int, **kwargs):
        super(SequenceClassificationLayer, self).__init__(**kwargs)
        self.feature_proj = nn.Linear(hidden_dim, proj_dim)
        self.attention_proj = nn.Linear(hidden_dim, proj_dim)
        self.cls_proj = nn.Linear(proj_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_proj(inputs)
        attention = self.attention_proj(inputs)
        attention = F.softmax(attention, dim=-2)
        seq_repr = torch.sum(attention * features, dim=-2)
        logits = self.cls_proj(seq_repr)
        return logits


class TokenClassificationLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, **kwargs):
        super(TokenClassificationLayer, self).__init__(**kwargs)
        self.cls_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.cls_layer(inputs)
        return logits


class MultilayerClassifier(nn.Module):
    def __init__(self, input_shape: tuple, hidden_dims: list, num_classes: int, **kwargs):
        super(MultilayerClassifier, self).__init__()
        self.classifer = nn.ModuleList([])
        self.classifer.append(nn.Linear(input_shape[-1], hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.classifer.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.classifer.append(nn.ReLU())
        self.classifer.append(nn.Linear(hidden_dims[-1], num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.mean(dim=1).mean(dim=1)
        for layer in self.classifer:
            x = layer(x)
        return x


class UncompressedMultilayerClassifier(nn.Module):
    def __init__(self, input_shape: tuple, hidden_dims: list, num_classes: int, **kwargs):
        super(UncompressedMultilayerClassifier, self).__init__()
        self.classifer = nn.ModuleList([])
        self.classifer.append(nn.Linear(np.prod(input_shape), hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.classifer.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.classifer.append(nn.ReLU())
        self.classifer.append(nn.Linear(hidden_dims[-1], num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.view(inputs.shape[0], -1)
        for layer in self.classifer:
            x = layer(x)
        return x


class BasicClassifier(nn.Module):
    def __init__(self, input_shape: tuple, hidden_dims: list, num_classes: int, **kwargs):
        super(BasicClassifier, self).__init__()
        self.classifier = nn.ModuleList([])
        self.classifier.append(nn.Linear(input_shape[-1], hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.classifier.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.classifier.append(nn.ReLU())
        self.classifier.append(nn.Linear(hidden_dims[-1], num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.classifier:
            x = layer(x)
        return x


class StandardClassifier(nn.Module):
    def __init__(self, input_shape: tuple, num_classes: int, **kwargs):
        super(StandardClassifier, self).__init__()
        self.classifer = nn.Linear(input_shape[-1], num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifer(inputs.reshape(inputs.shape[0], -1, inputs.shape[-1]).mean(dim=1))
