import logging
from typing import Tuple, List

from numpy import ndarray, zeros
from torch.utils.data import Dataset

from chowder.data import LabelDict, DataDict, SlideID


class DatasetLengthMismatchError(Exception):
    pass


logger = logging.getLogger(__name__)


class ChowderDataset(Dataset):
    def __init__(self, label_dict: LabelDict, data_dict: DataDict, num_tiles: int, num_features: int):
        """ Dataset object to be used for training.

        IMPORTANT: The slide data is zero padded to obtain an homogeneous number of tiles for every slide.

        :param label_dict: A dict containing labels indexed by slide id
        :param data_dict: A dict containing data fetchers indexed by slide id
        """

        self._label_dict = label_dict
        self._data_dict = data_dict
        self._num_tiles = num_tiles
        self._num_features = num_features

        if set(label_dict.keys()) != set(data_dict.keys()):
            raise DatasetLengthMismatchError('The data and labels dict do not contain the same IDs as keys.')

        self._slide_ids = list(label_dict.keys())

    def __len__(self) -> int:
        return len(self._label_dict.keys())

    def __getitem__(self, index: int) -> Tuple[ndarray, int]:
        data_placeholder = zeros([self._num_tiles, self._num_features]).astype('float32')

        slide_id = self._slide_ids[index]
        array_fetcher = self._data_dict[slide_id]
        try:
            slide_array: ndarray = array_fetcher()[:, 3:].astype('float32')  # The three first numbers are slide headers
        except (IOError, ValueError):
            logger.error(f"The slide data with corresponding ID could not be read from disk: {slide_id}, "
                         f"exiting application.")
            raise SystemExit()
        slide_shape = slide_array.shape
        data_placeholder[:slide_shape[0], :slide_shape[1]] = slide_array
        label = self._label_dict[slide_id]
        return data_placeholder, label.value

    @property
    def id_list(self) -> List[SlideID]:
        """ The list of slide ids contained in the dataset, which indicates the ordering of the slides
        :return: A list of slide ids
        """
        return self._slide_ids
