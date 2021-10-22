""" Neural network training and evaluation utilities. """

# Author: François Joubert <fxa.joubert@gmail.com>
# License: MIT
import logging
from dataclasses import dataclass
from typing import cast

import torch
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingParams:
    device: str
    log_interval: int
    tb_writer: SummaryWriter
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
          loss_func: nn.Module, epoch: int) -> None:
    """ Training routine for one epoch that goes through the dataset and performs the loss optimisation over batches
    of data.

    :param training_params: A structure containing parameters for the training
    :param model: The network model
    :param train_loader: A data loader containing the training data
    :param optimizer: The optimisation algorithm implementation
    :param loss_func: The loss function module
    :param epoch: The current epoch
    :return: Nothing
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

    training_params.tb_writer.add_scalar("%s_loss" % 'train', losses_avg_meter.average, epoch)


def evaluate(training_params: TrainingParams, model: nn.Module, test_loader: DataLoader, loss_func: nn.Module,
             epoch: int) -> float:
    """ Evaluation routine that goes through the testing data and evaluate a previously trained model. It computes the
    roc metric
    :param training_params: A structure containing parameters for the training
    :param model: The network model
    :param test_loader: A data loader containing the testing data
    :param loss_func: The loss function module
    :param epoch: The current epoch
    :return: roc_metric: the computed AUC score
    """
    losses_avg_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(training_params.device), target.to(training_params.device)
            output = model(data)
            test_loss = torch.sum(loss_func(output, target))
            losses_avg_meter.update(test_loss.item(), data.size(0))

    _, index = torch.max(output, dim=1)

    roc_metric = roc_auc_score(target.squeeze(0).numpy(), index.numpy())

    logger.info(
        "\nTest set: Average loss: {:.4f}, AUC: {:.4f}\n".format(
            losses_avg_meter.average, roc_metric
        )
    )

    training_params.tb_writer.add_scalar("%s_loss" % 'test', losses_avg_meter.average, epoch)
    training_params.tb_writer.add_scalar("%s_acc" % 'test', roc_metric, epoch)
    return roc_metric
