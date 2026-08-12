"""Minimal import and forward-pass regression test."""

import unittest

import torch

from src.model_load import resnet50


class ModelSmokeTest(unittest.TestCase):
    def test_resnet50_forward_shape(self) -> None:
        model = resnet50(dropout=0.1).eval()
        inputs = torch.randn(2, 3, 64, 64)

        with torch.inference_mode():
            predictions, encodings = model(inputs)

        self.assertEqual(tuple(predictions.shape), (2, 1))
        self.assertEqual(encodings.shape[0], 2)
        self.assertEqual(encodings.ndim, 2)


if __name__ == "__main__":
    unittest.main()
