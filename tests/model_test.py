from unittest import TestCase

from torch import randn, float32, max as tmax

from chowder.model import BaselineModel, ChowderModel


class ModelTest(TestCase):
    def setUp(self) -> None:
        self.num_features_seq = [128, 256, 2048]
        self.num_channels = 50
        self.batch_size = 10

    def test_baseline_output_correct_size_and_range(self):
        for num_features in self.num_features_seq:
            model = BaselineModel(num_features)
            input_data = randn(self.batch_size, self.num_channels, num_features, dtype=float32)

            out = model(input_data)
            self.assertEqual(out.shape, (self.batch_size, 2))  # the output is a binary prediction
            max_value = tmax(out)
            self.assertTrue(max_value < 0.0)  # logsoftmax outputs negative values

    def test_chowder_output_correct_size_and_range(self):
        r_values = [1, 5, 10]
        for num_features in self.num_features_seq:
            for r in r_values:
                model = ChowderModel(num_features, r)

                input_data = randn(self.batch_size, self.num_channels, num_features, dtype=float32)
                out = model(input_data)
                self.assertEqual(out.shape, (self.batch_size, 2))  # the output is a binary prediction
                max_value = tmax(out)
                self.assertTrue(max_value < 0.0)  # logsoftmax outputs negative values
