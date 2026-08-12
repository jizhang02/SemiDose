"""Run a data-free CPU forward pass through the SemiDose regressor."""

import torch

from src.model_load import resnet50


def main() -> None:
    torch.manual_seed(42)
    model = resnet50(dropout=0.1).eval()
    synthetic_batch = torch.randn(2, 3, 256, 256)

    with torch.inference_mode():
        dose_prediction, encoding = model(synthetic_batch)

    print(f"input:      {tuple(synthetic_batch.shape)}")
    print(f"prediction: {tuple(dose_prediction.shape)}")
    print(f"encoding:   {tuple(encoding.shape)}")

    assert dose_prediction.shape == (2, 1)
    assert encoding.ndim == 2 and encoding.shape[0] == 2


if __name__ == "__main__":
    main()
