""" Data loading and manipulation utilities. """

# Author: François Joubert <fxa.joubert@gmail.com>
# License: MIT

import csv
import logging
import re
from enum import Enum
from functools import partial
from pathlib import Path
from typing import NewType, Dict, Optional, Callable, Sequence, List

from numpy import ndarray, load

SlideID = NewType('SlideID', int)


class Label(Enum):
    HEALTHY = 0
    CANCER = 1


LabelDict = Dict[SlideID, Label]
DataDict = Dict[SlideID, Callable[[], ndarray]]

ID_HEADER = 'ID'
LABEL_HEADER = 'Target'

logger = logging.getLogger(__name__)


class IllshapedFileException(Exception):
    pass


def load_labels_as_dict(labels_filepath: Path) -> Optional[LabelDict]:
    """ Load the labels from an input csv file respecting the following format:

    ---
    ID,Target
    id_0,0
    etc.
    ---

    :param labels_filepath: a path to a csv input file containing slide ids and corresponding labels
    :return: a dict which maps a slide ID to its label
    """
    labels_dict: Dict[SlideID, Label] = {}
    try:
        with open(str(labels_filepath), newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames and ID_HEADER in reader.fieldnames and LABEL_HEADER in reader.fieldnames:
                for row in reader:
                    slide_id = find_slide_id_from_str(row[ID_HEADER])
                    label = Label(int(float(row[LABEL_HEADER])))
                    if slide_id is not None:
                        labels_dict[slide_id] = label
                    else:
                        logger.warning(f'Wrong ID in row: {row}, label skipped.')
            else:
                raise IllshapedFileException('The file does not contain the expected headers, loading failed.')

    except (FileNotFoundError, IOError, IllshapedFileException) as er:
        logger.error(f'The input file could not be read: {er}')
        return None

    return labels_dict


def load_slide_data_as_dict(data_filepaths: Sequence[Path]) -> DataDict:
    """ Load slide data contained in a sequence of files and gather it in a dictionary. The data is not fetched
    from disk, the dictionary maps slide IDs to functions that load the data from disk upon call.

    :param data_filepaths: the path of a file containing slide data
    :return: A dict mapping a slide ID to a partial function which loads a numpy array upon call.
    """
    slide_data_dict: Dict[SlideID, Callable[[], ndarray]] = {}

    for data_filepath in data_filepaths:
        data_fetcher = partial(load, data_filepath)
        slide_id = find_slide_id_from_str(data_filepath.stem)
        if slide_id is not None:
            slide_data_dict[slide_id] = data_fetcher
        else:
            logger.warning(f'Missing slide id for filepath: {data_filepath}')

    return slide_data_dict


def find_slide_id_from_str(slide_id_str: str) -> Optional[SlideID]:
    """ Match a str with a slide ID respecting the format 'ID_xxx_any_other_thing_here'.

    :param slide_id_str: a string containing a slide ID
    :return: A slide ID if the string matches, None otherwise
    """
    slide_id_regex = re.compile(r'\d{3}')
    re_search = slide_id_regex.search(slide_id_str)
    if re_search:
        return SlideID(int(re_search.group()))
    else:
        return None


def save_prediction_on_disk(filepath: Path, predictions: ndarray, slide_ids_list: List[SlideID]) -> None:
    """ Store evaluation results in a csv file.

    IMPORTANT: The prediction array and the list should point to the same underlying data, i.e the order of
    the slides should correspond.

    :param filepath: The path to the csv file to be written to
    :param predictions: The array of prediction output by the model
    :param slide_ids_list: The corresponding slide IDs
    :return: Nothing
    """

    if len(predictions) != len(slide_ids_list):
        logger.error('The predictions array and the slide IDs list do not match. Aborting... ')
        return None
    try:
        with open(str(filepath), 'w', newline='') as csvfile:
            fieldnames = [ID_HEADER, LABEL_HEADER]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for index, prediction in enumerate(predictions):
                slide_id = slide_ids_list[index]
                writer.writerow({ID_HEADER: f'ID_{slide_id:03d}', LABEL_HEADER: float(prediction)})
    except OSError as er:
        logger.error(f'The test output csv file could not be written: {er}')
