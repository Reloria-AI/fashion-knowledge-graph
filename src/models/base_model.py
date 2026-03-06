"""
Base model abstractions for ML components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseModel(ABC):
    """Common base class for all model wrappers."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[Any] = None
        self._is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights/resources into memory."""

    def ensure_model_loaded(self) -> None:
        """Lazy-load the model if it is not loaded yet."""
        if not self._is_loaded:
            self.load_model()


class BaseEmbeddingModel(BaseModel, ABC):
    """Abstract base class for embedding models."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        super().__init__(model_name=model_name, device=device)
        self.embedding_dim: Optional[int] = None

    @abstractmethod
    def get_embedding(
        self,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        type: str = "text",
    ) -> list[float]:
        """Generate an embedding for text or image input."""


class BaseSegmentationModel(BaseModel, ABC):
    """Abstract base class for segmentation models."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        id2label: Optional[Dict[int, str]] = None,
    ):
        super().__init__(model_name=model_name, device=device)
        self.id2label: Dict[int, str] = id2label or {
            0: "Background",
            3: "Sunglasses",
            4: "Upper-clothes",
            5: "Skirt",
            6: "Pants",
            7: "Dress",
            8: "Belt",
            16: "Bag",
            17: "Scarf",
        }
        self.label2id: Dict[str, int] = {v: k for k, v in self.id2label.items()}

    def get_id2label(self) -> Dict[int, str]:
        """Return label-id mapping."""
        return self.id2label
