"""
Grad-CAM for EfficientNetB0: forward hook on last conv layer, heatmap overlay.
"""
from pathlib import Path
import numpy as np
import torch
import cv2
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD


class GradCAM:
    """
    Grad-CAM: weight activations by mean gradient and sum to get 2D heatmap.
    Target: last conv layer of EfficientNetB0 (features[8]).
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hook_handles = []

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _register_hooks(self):
        self._hook_handles.append(self.target_layer.register_forward_hook(self._save_activations))
        self._hook_handles.append(self.target_layer.register_full_backward_hook(self._save_gradients))

    def _remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def __call__(self, x: torch.Tensor, class_idx=None):
        """
        x: (1, 3, H, W) tensor. class_idx: which class to backprop (default: predicted).
        Returns: heatmap (H, W) numpy, and overlay (H, W, 3) RGB numpy in [0, 255].
        """
        self.model.eval()
        self._register_hooks()
        try:
            out = self.model(x)
            self.model.zero_grad()
            if out.dim() == 1:
                out = out.unsqueeze(0)
            # Binary: single logit; backprop w.r.t. that (explains "cataract" evidence)
            if out.shape[1] == 1:
                out[0, 0].backward()
            else:
                class_idx = out.argmax(dim=1).item() if class_idx is None else class_idx
                out[0, class_idx].backward()
        finally:
            self._remove_hooks()

        if self.activations is None or self.gradients is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE)), np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        # weights = mean of gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3))  # (1, C)
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.activations).sum(dim=1).squeeze(0)  # (H', W')
        cam = torch.relu(cam).cpu().numpy()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heatmap = (cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET))
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        # Overlay on input image (denormalize x for display)
        img = x[0].cpu().permute(1, 2, 0).numpy()
        img = img * np.array(IMAGE_STD) + np.array(IMAGE_MEAN)
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        overlay = (0.5 * img + 0.5 * heatmap).astype(np.uint8)
        return cam, overlay


def get_efficientnet_target_layer(model):
    """Return the last conv layer of EfficientNetB0 (features[8])."""
    # efficientnet_b0.features is a Sequential; last block is index 8
    if hasattr(model, "backbone") and hasattr(model.backbone, "features"):
        features = model.backbone.features
    elif hasattr(model, "features"):
        features = model.features
    else:
        raise AttributeError("Model has no .features or .backbone.features")
    return features[8]


def run_gradcam(model, x: torch.Tensor, device=None):
    """
    Run Grad-CAM for EfficientNetB0 on input x (1, 3, H, W).
    Returns overlay image (H, W, 3) numpy uint8.
    """
    if device is None:
        device = next(model.parameters()).device
    x = x.to(device)
    target_layer = get_efficientnet_target_layer(model)
    gradcam = GradCAM(model, target_layer)
    cam, overlay = gradcam(x)
    return overlay
