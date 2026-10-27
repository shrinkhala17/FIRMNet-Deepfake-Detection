"""
gradcam_utils.py — Grad-CAM visualization for deepfake detection.

Implements Grad-CAM (Gradient-weighted Class Activation Mapping) for
MobileNetV3-Small to highlight image regions influencing the FAKE/REAL
prediction.

No external dependencies beyond PyTorch and PIL/matplotlib.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization", ICCV 2017.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


class GradCAM:
    """Grad-CAM for any CNN model.

    Args:
        model: PyTorch model in eval mode.
        target_layer: The convolutional layer to hook into.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: [1, C, H, W] preprocessed image tensor.
            target_class: Class index to explain. If None, uses predicted class.

        Returns:
            heatmap: [H, W] numpy array in [0, 1] — the Grad-CAM activation map.
            predicted_class: int — the model's predicted class index.
            probabilities: [num_classes] numpy array — softmax probabilities.
        """
        self.model.eval()

        # Enable gradients for Grad-CAM (even though we're in eval mode)
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0].detach().cpu().numpy()
        predicted_class = output.argmax(dim=1).item()

        if target_class is None:
            target_class = predicted_class

        # Backward pass for the target class
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Grad-CAM computation
        gradients = self.gradients[0]    # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Global average pool the gradients
        weights = gradients.mean(dim=(1, 2))  # [C]

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=activations.dtype,
                          device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — only positive contributions
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, predicted_class, probabilities

    def cleanup(self):
        """Remove hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def generate_gradcam_figure(model, input_tensor, original_image,
                            target_layer, class_names=None):
    """Generate a Grad-CAM visualization figure.

    Args:
        model: Trained PyTorch model.
        input_tensor: [1, C, H, W] preprocessed tensor.
        original_image: PIL Image (original, before preprocessing).
        target_layer: nn.Module — target conv layer.
        class_names: dict mapping class index to name.

    Returns:
        figure: matplotlib Figure with the heatmap overlay.
        result: dict with predicted_class, probabilities, heatmap.
    """
    if class_names is None:
        class_names = {0: "FAKE", 1: "REAL"}

    # Generate Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    try:
        heatmap, pred_class, probs = grad_cam.generate(input_tensor)
    finally:
        grad_cam.cleanup()

    # Resize heatmap to match original image
    img_array = np.array(original_image.convert("RGB").resize((224, 224)))
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (img_array.shape[1], img_array.shape[0]),
            Image.BILINEAR
        )
    ) / 255.0

    # Create colored heatmap overlay
    colormap = cm.jet(heatmap_resized)[:, :, :3]  # Remove alpha
    overlay = (0.5 * img_array / 255.0 + 0.5 * colormap)
    overlay = np.clip(overlay, 0, 1)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image", fontsize=12, color="white", pad=10)
    axes[0].axis("off")

    # Grad-CAM heatmap
    im = axes[1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12, color="white", pad=10)
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(overlay)
    pred_name = class_names.get(pred_class, str(pred_class))
    conf = probs[pred_class]
    axes[2].set_title(
        f"Overlay: {pred_name} ({conf:.1%})",
        fontsize=12, color="white", pad=10,
    )
    axes[2].axis("off")

    fig.patch.set_facecolor("#0a0e1a")
    for ax in axes:
        ax.set_facecolor("#0a0e1a")

    fig.tight_layout()

    result = {
        "predicted_class": pred_class,
        "predicted_name": pred_name,
        "probabilities": probs,
        "heatmap": heatmap,
    }

    return fig, result


def get_mobilenet_target_layer(model):
    """Get the last convolutional layer of MobileNetV3-Small for Grad-CAM."""
    # The last convolutional block in MobileNetV3-Small features
    # features[-1] is the final InvertedResidual + final conv block
    return model.features[-1]


def get_firmnet_target_layer(model):
    """Get the last convolutional stage of FIRMNet for Grad-CAM."""
    # stage6 is the deepest convolutional stage before the head
    return model.stage6
