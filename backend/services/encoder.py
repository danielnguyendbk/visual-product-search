"""Pretrained ResNet50 features for single images and image batches."""

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Encoder:
    """Load the model once and reuse it for inference."""

    model_name = "resnet50"
    embedding_dimension = 2048
    weights = ResNet50_Weights.IMAGENET1K_V2

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocess = self.weights.transforms()
        self.model = resnet50(weights=self.weights)
        self.model.fc = torch.nn.Identity()
        self.model.requires_grad_(False)
        self.model.float().to(self.device)
        self.model.eval()

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Return one normalized float32 vector with shape (2048,)."""
        return self.encode_batch([image])[0]

    @torch.inference_mode()
    def encode_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return normalized float32 vectors in the input image order."""
        if not images:
            raise ValueError("An image batch must not be empty.")

        inputs = []
        for index, image in enumerate(images):
            try:
                inputs.append(self.preprocess(image.convert("RGB")))
            except Exception as error:
                raise ValueError(f"Cannot preprocess batch image {index}: {error}") from error

        batch = torch.stack(inputs).to(self.device)
        features = self.model(batch).to(dtype=torch.float32)
        if features.shape != (len(images), self.embedding_dimension):
            raise ValueError(f"Unexpected feature shape: {tuple(features.shape)}")
        if not torch.isfinite(features).all():
            raise ValueError("ResNet50 produced NaN or Inf features.")

        norms = torch.linalg.vector_norm(features, ord=2, dim=1, keepdim=True)
        if not torch.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("Cannot L2 normalize a non-finite or zero feature vector.")
        return (features / norms).cpu().numpy()


@lru_cache(maxsize=1)
def get_encoder() -> ResNet50Encoder:
    """Create the shared process encoder only when it is first needed."""
    return ResNet50Encoder()


def encode_image(image: Image.Image) -> np.ndarray:
    """Keep the existing single-image service interface."""
    return get_encoder().encode_image(image)


def encode_batch(images: Sequence[Image.Image]) -> np.ndarray:
    """Encode a batch using the same shared model as encode_image()."""
    return get_encoder().encode_batch(images)
