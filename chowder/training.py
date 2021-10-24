""" Neural network training and evaluation utilities. """

# Author: François Joubert <fxa.joubert@gmail.com>
# License: MIT

import logging
from dataclasses import dataclass
from typing import cast, Tuple

import torch
from numpy import ndarray
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingParams:
    device: str
    log_interval: int
    num_epochs: int
    eval_interval: int
    learning_rate: float


class AverageMeter:
    """ Keep track of a given float quantity average value. """

    def __init__(self) -> None:
        self._current_val: float = 0
        self._average: float = 0
        self._sum: float = 0
        self._count: float = 0

    def update(self, val: float, n: int = 1) -> None:
        self._current_val = val
        self._sum += val * n
        self._count += n
        self._average = self._sum / self._count if self._count != 0 else 0

    @property
    def average(self) -> float:
        return self._average


def train(training_params: TrainingParams, model: nn.Module, train_loader: DataLoader, optimizer: Optimizer,
          loss_func: nn.Module, epoch: int) -> float:
    """ Training routine for one epoch that goes through the dataset and performs the loss optimisation over batches
    of data.

    :param training_params: A structure containing parameters for the training
    :param model: The network model
    :param train_loader: A data loader containing the training data
    :param optimizer: The optimisation algorithm implementation
    :param loss_func: The loss function module
    :param epoch: The current epoch
    :return: The average loss on the training step
    """
    model.train()
    losses_avg_meter = AverageMeter()

    batch_size: int = cast(int, train_loader.batch_size)  # Batch size is specified in our case
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(training_params.device), target.to(training_params.device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        losses_avg_meter.update(loss.item(), data.size(0))
        if batch_idx % training_params.log_interval == 0 and batch_idx != 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * batch_size,
                    len(train_loader) * batch_size,
                    100.0 * batch_idx / len(train_loader),
                    losses_avg_meter.average,
                )
            )
    return losses_avg_meter.average


def evaluate(training_params: TrainingParams, model: nn.Module, test_loader: DataLoader, loss_func: nn.Module) -> \
        Tuple[float, float, ndarray]:
    """ Evaluation routine that goes through the testing data and evaluate a previously trained model. It computes the
    roc metric
    :param training_params: A structure containing parameters for the training
    :param model: The network model
    :param test_loader: A data loader containing the testing data
    :param loss_func: The loss function module
    :param epoch: The current epoch
    :return:
      average_loss: the average loss over the test set
      roc_metric: the computed AUC score
      predictions: the predictions over the test set
    """
    losses_avg_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(training_params.device), target.to(training_params.device)
            output = model(data)
            test_loss = torch.sum(loss_func(output, target))
            losses_avg_meter.update(test_loss.item(), data.size(0))

    _, output_classes = torch.max(output, dim=1)
    prediction = output_classes.numpy()

    auc_score = roc_auc_score(target.squeeze(0).numpy(), prediction)
    average_loss = losses_avg_meter.average
    logger.info(
        "\nTest set: Average loss: {:.4f}, AUC: {:.4f}\n".format(
            average_loss, auc_score
        )
    )

    return average_loss, auc_score, prediction
