"""
Module 4 - Payment Image Detection
Explainability: Grad-CAM over EfficientNet-B0's last conv block.
Produces the tampering-localization heatmap referenced in the module's
Output contract (Section 5.4) and Explainability Layer (Section 7).

Note: Grad-CAM requires a scalar target (the classifier_head logit in
EfficientNetB0FeatureExtractor). If the head hasn't been fine-tuned on
fraud labels, the heatmap highlights generic salient regions instead of
tampering-specific ones — fine-tune classifier_head jointly or via a
small auxiliary training pass before relying on this for reports.
"""

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from vision.efficientnet_extractor import build_transform


class _LogitWrapper(torch.nn.Module):
    """pytorch-grad-cam expects a model whose forward() returns logits directly."""

    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, x):
        embedding = self.feature_extractor(x)
        return self.feature_extractor.classifier_head(embedding)


class PaymentGradCAM:
    def __init__(self, feature_extractor, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.wrapped_model = _LogitWrapper(feature_extractor).to(self.device).eval()
        self.target_layer = feature_extractor.target_conv_layer
        self.transform = build_transform()
        self.cam = GradCAM(model=self.wrapped_model, target_layers=[self.target_layer])

    def generate(self, image_path: str, out_path: str, target_class: int = 0) -> str:
        raw_img = cv2.imread(image_path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        raw_img_resized = cv2.resize(raw_img, (224, 224)).astype(np.float32) / 255.0

        from PIL import Image

        pil_img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)[0]

        visualization = show_cam_on_image(raw_img_resized, grayscale_cam, use_rgb=True)
        cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        return out_path
