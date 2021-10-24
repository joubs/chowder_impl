from copy import deepcopy
from random import randint
from typing import Tuple
from unittest import TestCase

from torch import nn, Tensor, randn, optim, equal
from torch.utils.data import Dataset, DataLoader

from chowder.training import TrainingParams, train, evaluate


class Rand1DDataset(Dataset):
    def __init__(self, set_size: int, signal_length: int) -> None:
        super().__init__()
        self._size = set_size
        self._length = signal_length

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        return randn(self._length), randint(0, 1)


class TrainingTest(TestCase):
    def setUp(self) -> None:
        self._data_length = 100
        self._training_params = TrainingParams(device='cpu', log_interval=5, num_epochs=10,
                                               eval_interval=1, learning_rate=1e-3)
        self._model = nn.Sequential(
            nn.Linear(self._data_length, 2),
            nn.LogSoftmax(dim=1))
        self._dataset = Rand1DDataset(200, self._data_length)
        self._dataloader = DataLoader(self._dataset, batch_size=10)
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._training_params.learning_rate)
        self._loss_fn = nn.NLLLoss()

    def test_training_iteration_change_model_weights(self):
        step_0_parameters = deepcopy(self._model.state_dict())
        train(self._training_params, self._model, self._dataloader, self._optimizer, self._loss_fn, 1)
        step_1_parameters = deepcopy(self._model.state_dict())

        for key, val in step_0_parameters.items():
            self.assertFalse(equal(val, step_1_parameters[key]))

    def test_evaluation_keeps_model_untouched(self):
        step_0_parameters = deepcopy(self._model.state_dict())
        evaluate(self._training_params, self._model, self._dataloader, self._loss_fn)
        step_1_parameters = deepcopy(self._model.state_dict())

        for key, val in step_0_parameters.items():
            self.assertTrue(equal(val, step_1_parameters[key]))

