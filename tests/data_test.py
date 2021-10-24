from pathlib import Path
from unittest import TestCase

from chowder.data import load_labels_as_dict, find_slide_id_from_str, Label, load_slide_data_as_dict, SlideID

RESOURCE_FOLDER = Path(__file__).parent / 'resources'

TEST_LABELS_CSV = RESOURCE_FOLDER / 'test_labels.csv'
ILLSHAPED_LABELS_CSV = RESOURCE_FOLDER / 'illshaped_labels.csv'

ID_18_SLIDE_DATA = RESOURCE_FOLDER / 'ID_018.npy'


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
        self.assertTrue(1 in labels_dict.keys())
        self.assertEqual(labels_dict[1], Label.HEALTHY)

    # Slide data

    def test_good_npy_array_file_loads_okay(self):
        slide_data_dict = load_slide_data_as_dict([ID_18_SLIDE_DATA])
        self.assertTrue(18 in slide_data_dict.keys())

        # Check data is fetched without raising
        data_fetcher = slide_data_dict[SlideID(18)]
        data_fetcher()

    def test_missing_file_implies_no_entry_in_dict(self):
        slide_data_dict = load_slide_data_as_dict([Path('path/to/data')])
        self.assertTrue(len(slide_data_dict.keys()) == 0)

    # Slide ids

    def test_wrong_slide_id_in_filename_returns_none(self):
        default_filepath = Path('ID_010.npy')
        annotated_filepath = Path('ID_011_annotated.npy')
        wrong_filepath = Path('_wrong_file.npy')

        self.assertEqual(10, find_slide_id_from_str(default_filepath.stem))
        self.assertEqual(11, find_slide_id_from_str(annotated_filepath.stem))
        self.assertIsNone(find_slide_id_from_str(wrong_filepath.stem))
