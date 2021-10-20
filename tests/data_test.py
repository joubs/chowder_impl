from pathlib import Path
from unittest import TestCase

from chowder.data import load_labels_as_dict, find_slide_id_from_filepath, Label

RESOURCE_FOLDER = Path(__file__).parent / 'resources'

TEST_LABELS_CSV = RESOURCE_FOLDER / 'test_labels.csv'
ILLSHAPED_LABELS_CSV = RESOURCE_FOLDER / 'illshaped_labels.csv'


class DataTest(TestCase):

    # Labels

    def test_nonexistent_labels_file_returns_none(self):
        nonexistent_file = Path(r'/non/existent/file')
        labels_dict = load_labels_as_dict(nonexistent_file)
        self.assertIsNone(labels_dict)

    def test_illshaped_labels_file_returns_none(self):
        labels_dict = load_labels_as_dict(ILLSHAPED_LABELS_CSV)
        self.assertIsNone(labels_dict)

    def test_good_labels_file_loads_ok(self):
        labels_dict = load_labels_as_dict(TEST_LABELS_CSV)
        self.assertIsNotNone(labels_dict)
        self.assertTrue('ID_001' in labels_dict.keys())
        self.assertEqual(labels_dict['ID_001'], Label.HEALTHY)

    # Slide ids

    def test_wrong_slide_id_in_filename_returns_none(self):
        default_filepath = Path('ID_010.npy')
        annotated_filepath = Path('ID_011_annotated.npy')
        wrong_filepath = Path('_wrong_file.npy')

        self.assertEqual('ID_010', find_slide_id_from_filepath(default_filepath))
        self.assertEqual('ID_011_annotated', find_slide_id_from_filepath(annotated_filepath))
        self.assertIsNone(find_slide_id_from_filepath(wrong_filepath))
