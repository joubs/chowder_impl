""" Data loading and manipulation utilities. """

# Author: François Joubert <fxa.joubert@gmail.com>
# License: MIT

import csv
import logging
import re
from enum import Enum
from functools import partial
from pathlib import Path
from typing import NewType, Dict, Optional, Callable, Sequence

from numpy import ndarray, load

SlideID = NewType('SlideID', str)


class Label(Enum):
    HEALTHY = 0
    CANCER = 1


ID_HEADER = 'ID'
LABEL_HEADER = 'Target'

logger = logging.getLogger(__name__)


class IllshapedFileException(Exception):
    pass


def load_labels_as_dict(labels_filepath: Path) -> Optional[Dict[SlideID, Label]]:
    """ Load the labels from an input csv file respecting the following format:

    ---
    ID,Target
    id_0,0
    etc.
    ---

    :param labels_filepath: a path to a csv input file containing slide ids and corresponding labels
    :return: a dict which maps an id to its label
    """
    labels_dict: Dict[SlideID, Label] = {}
    try:
        with open(str(labels_filepath), newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames and ID_HEADER in reader.fieldnames and LABEL_HEADER in reader.fieldnames:
                for row in reader:
                    slide_id = SlideID(row[ID_HEADER])
                    label = Label(int(float(row[LABEL_HEADER])))  # todo: This is ugly
                    labels_dict[slide_id] = label
            else:
                raise IllshapedFileException('The file does not contain the expected headers, loading failed.')

    except (FileNotFoundError, IOError, IllshapedFileException) as er:
        logger.error(f'The input file could not be read: {er}')
        return None

    return labels_dict


def load_slide_data_as_dict(data_filepaths: Sequence[Path]) -> Dict[SlideID, Callable[[], ndarray]]:
    """ load slide data contained in a sequence of files and gather it in a dictionary. The data is not fetched
    from disk, the dictionary maps slide IDs to functions that load the data from disk upon call.

    :param data_filepaths: the path of a file containing slide data
    :return: A dict mapping a slide ID to a partial function which loads a numpy array upon call.
    """
    slide_data_dict: Dict[SlideID, Callable[[], ndarray]] = {}

    for data_filepath in data_filepaths:
        data_fetcher = partial(load, data_filepath)
        slide_id = find_slide_id_from_filepath(data_filepath)
        if slide_id:
            slide_data_dict[slide_id] = data_fetcher
        else:
            logger.warning(f'Missing slide id for filepath: {data_filepath}')

    return slide_data_dict


def find_slide_id_from_filepath(slide_data_filepath: Path) -> Optional[SlideID]:
    """ Match a filename with a slide ID respecting the format 'ID_xxx_any_other_thing_here'.

    :param slide_data_filepath: the path of a file containing slide data
    :return: A slide ID if the filename matches, None otherwise
    """
    slide_id_regex = re.compile(r'ID_\d{3}.*')
    filename = slide_data_filepath.stem
    if slide_id_regex.match(filename):
        return SlideID(filename)
    else:
        return None
