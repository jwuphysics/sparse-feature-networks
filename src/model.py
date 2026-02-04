import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18TopK(nn.Module):
    def __init__(self, k: int = 2, n_out: int = 1, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.resnet = resnet18(weights=None)

        self.k = k

        # cache trunk so we don't rebuild modules every forward
        self.trunk = nn.Sequential(*list(self.resnet.children())[:-1])

        n_fc_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(n_fc_in, n_out)

    def sparse_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        features = torch.flatten(features, 1)
        features = nn.functional.relu(features)

        topk_values, topk_indices = torch.topk(features, k=self.k, dim=1)
        sparse_features = torch.zeros_like(features)
        sparse_features.scatter_(1, topk_indices, topk_values)
        return sparse_features

    def forward(self, x):
        sparse_features = self.sparse_features(x)
        return self.resnet.fc(sparse_features)
