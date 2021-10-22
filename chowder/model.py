""" Models implementations """

# Author: François Joubert <fxa.joubert@gmail.com>
# License: MIT

import torch.nn as nn
from torch import Tensor


class BaselineModel(nn.Module):
    """ Baseline model to be compared with CHOWDER implementation. Provided the resnet50 features given as input,
    it performs a pooling over the tiles axis and then trains a binary classifier.

    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self._pooling = nn.AdaptiveMaxPool2d(output_size=(1, num_features))
        self._classifier = nn.Sequential(
            nn.Linear(num_features, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._pooling(x).squeeze(1)
        x = self._classifier(x)
        return x


class ChowderModel(nn.Module):
    pass
