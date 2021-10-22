from typing import Tuple

from numpy import ndarray, zeros
from torch.utils.data import Dataset

from chowder.data import LabelDict, DataDict


class DatasetLengthMismatchError(Exception):
    pass


class ChowderDataset(Dataset):
    def __init__(self, label_dict: LabelDict, data_dict: DataDict, num_tiles: int):
        """ Dataset object to be used for training.

        :param label_dict: A dict containing labels indexed by slide id
        :param data_dict: A dict containing data fetchers indexed by slide id
        """

        self._label_dict = label_dict
        self._data_dict = data_dict
        self._num_tiles = num_tiles

        if set(label_dict.keys()) != set(data_dict.keys()):
            raise DatasetLengthMismatchError('The data and labels dict do not contain the same IDs as keys.')

        self._slide_ids = list(label_dict.keys())

    def __len__(self) -> int:
        return len(self._label_dict.keys())

    def __getitem__(self, index: int) -> Tuple[ndarray, int]:
        data_placeholder = zeros([self._num_tiles, 2048]).astype('float32')

        slide_id = self._slide_ids[index]
        array_fetcher = self._data_dict[slide_id]
        slide_array: ndarray = array_fetcher()[:, 3:].astype('float32')  # The three first numbers are slide headers
        slide_shape = slide_array.shape
        data_placeholder[:slide_shape[0], :slide_shape[1]] = slide_array  # Todo: this should be tested
        label = self._label_dict[slide_id]
        return data_placeholder, label.value
