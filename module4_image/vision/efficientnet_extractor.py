"""
Module 4 - Payment Image Detection
Visual feature extraction using a pretrained, fine-tunable EfficientNet-B0.
Produces a 1280-dim embedding per screenshot that captures tampering /
splicing artifacts and logo authenticity cues (Section 5.4).

Also exposes the last conv feature map + model handle needed by
explainability/gradcam_explainer.py for tampering-localization heatmaps.
"""

from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class EfficientNetB0FeatureExtractor(nn.Module):
    """
    Wraps torchvision's EfficientNet-B0.
    - `embedding_dim` = 1280 (pre-classifier pooled features)
    - `forward` returns (embedding, fraud_logit) when `num_classes` is set,
      or just `embedding` when used purely as a feature extractor
      (fine_tune=False path used by the LightGBM fusion classifier).
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 1, freeze_backbone: bool = False):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Keep feature extractor (conv layers) + pooling, drop the ImageNet classifier.
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.embedding_dim = 1280

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        # Optional lightweight head for end-to-end fine-tuning as an
        # auxiliary loss; the primary fraud decision still comes from
        # the LightGBM/MLP fusion classifier downstream.
        self.classifier_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.embedding_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, return_logit: bool = False):
        feat_map = self.features(x)          # used by Grad-CAM (last conv layer)
        pooled = self.avgpool(feat_map)
        embedding = torch.flatten(pooled, 1)  # [B, 1280]

        if return_logit:
            logit = self.classifier_head(embedding)
            return embedding, logit, feat_map
        return embedding

    @property
    def target_conv_layer(self):
        """Layer to hook for Grad-CAM (last block of EfficientNet-B0 features)."""
        return self.features[-1]


class VisualEmbeddingService:
    """Convenience wrapper: image path in, numpy embedding out."""

    def __init__(self, device: str = None, checkpoint_path: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EfficientNetB0FeatureExtractor(pretrained=True).to(self.device)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval()
        self.transform = build_transform()
        if self.device.startswith("cuda"):
            print(f"[VisualEmbeddingService] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[VisualEmbeddingService] Using CPU (no CUDA GPU detected)")

    @torch.no_grad()
    def embed(self, image_path: str) -> Tuple["numpy.ndarray", "PIL.Image.Image"]:
        import numpy as np

        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        embedding = self.model(tensor)
        return embedding.squeeze(0).cpu().numpy(), img

    @torch.no_grad()
    def embed_batch(self, image_paths: list, batch_size: int = 32):
        """
        Batched inference -- much faster than calling embed() in a loop when
        a GPU is available, since the GPU processes the whole batch in one
        forward pass instead of one image at a time.
        Returns: dict {image_path: embedding (np.ndarray)}. Paths that fail
        to load are silently skipped (caller should diff against the input
        list to detect skips).
        """
        import numpy as np

        results = {}
        for start in range(0, len(image_paths), batch_size):
            chunk = image_paths[start:start + batch_size]
            tensors, valid_paths = [], []
            for p in chunk:
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(self.transform(img))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"[skip] {p}: {e}")
            if not tensors:
                continue
            batch = torch.stack(tensors).to(self.device)
            embeddings = self.model(batch).cpu().numpy()
            for p, emb in zip(valid_paths, embeddings):
                results[p] = emb
        return results


if __name__ == "__main__":
    import sys

    service = VisualEmbeddingService()
    emb, _ = service.embed(sys.argv[1])
    print("Embedding shape:", emb.shape)
