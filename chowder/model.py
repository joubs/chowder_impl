""" Models implementations """

# Author: François Joubert <fxa.joubert@gmail.com>
# License: MIT

import torch.nn as nn
from torch import Tensor, reshape, sort, sigmoid


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
    """ CHOWDER model implementation, described in the following paper: https://arxiv.org/pdf/1802.02212.pdf

    """

    def __init__(self, num_features: int, R: int) -> None:
        super().__init__()
        self._conv1 = nn.Conv1d(1, 1, kernel_size=num_features, padding=0)
        self._fc1 = nn.Linear(2 * R, 200)
        self._fc2 = nn.Linear(200, 100)

        self._classifier = nn.Sequential(
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

        self._minmax_indices = [x for x in range(R)] + [x for x in range(-1, -R - 1, -1)]

    def forward(self, x: Tensor) -> Tensor:
        # 1d Conv
        b, c, l_in = x.shape
        x = reshape(x, [b * c, 1, l_in])
        x = self._conv1(x)
        x = reshape(x, [b, c])

        # MinMax
        x, _ = sort(x, dim=1)
        x = x[:, self._minmax_indices]

        # FC layer
        x = sigmoid(self._fc1(x))
        x = sigmoid(self._fc2(x))

        # Classifier
        x = self._classifier(x)
        return x
