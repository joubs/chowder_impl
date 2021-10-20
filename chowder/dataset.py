from typing import Tuple

from numpy import ndarray
from torch.utils.data import Dataset

from chowder.data import LabelDict, DataDict, Label, SlideID


class DatasetLengthMismatchError(Exception):
    pass


class ChowderDataset(Dataset):
    def __init__(self, label_dict: LabelDict, data_dict: DataDict):
        """ Dataset object to be used for training.

        :param label_dict: A dict containing labels indexed by slide id
        :param data_dict: A dict containing data fetchers indexed by slide id
        """

        self._label_dict = label_dict
        self._data_dict = data_dict

        if set(label_dict.keys()) != set(data_dict.keys()):
            raise DatasetLengthMismatchError('The data and labels dict do not contain the same IDs as keys.')

    def __len__(self) -> int:
        return len(self._label_dict.keys())

    def __getitem__(self, index: int) -> Tuple[ndarray, Label]:
        """

        :param index:
        :return:
        """
        slide_id = SlideID(index)
        array_fetcher = self._data_dict[slide_id]
        slide_array = array_fetcher()[:, 3:]  # The three first numbers are slide headers
        label = self._label_dict[slide_id]
        return slide_array, label
