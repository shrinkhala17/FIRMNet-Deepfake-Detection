import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image


# Sentinel value returned when analysis cannot proceed.
_ZERO_RATIOS = {"low": 0.0, "mid": 0.0, "high": 0.0}


def _make_blank_figure():
    """Return a placeholder figure for cases where FFT cannot be computed."""
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.text(0.5, 0.5, "FFT unavailable", ha="center", va="center",
            fontsize=14, color="#888")
    ax.set_facecolor("#0a0e1a")
    fig.patch.set_facecolor("#0a0e1a")
    ax.axis("off")
    fig.tight_layout()
    return fig


def analyze_frequency(image):
    """
    Analyze the frequency components of an image
    using the Fast Fourier Transform (FFT).

    Computes the 2D FFT, produces a centered log-magnitude spectrum,
    and calculates low-, mid-, and high-frequency energy ratios using
    circular (radial) frequency masks.

    Band definitions (based on radial distance from centre, normalised
    to half the image size):
        - Low:  distance <= 0.15  (coarse structure, uniform regions)
        - Mid:  0.15 < distance <= 0.50  (textures, edges)
        - High: distance > 0.50  (fine detail, noise, compression artefacts)

    These frequency ratios are descriptive image statistics.  They are
    **not** standalone proof of a deepfake — they are shown for
    educational and research purposes alongside the model's prediction.

    Args:
        image: PIL Image (any mode — will be converted to grayscale).

    Returns:
        figure: matplotlib Figure showing the FFT log-magnitude spectrum.
        frequency_ratios: dict with keys 'low', 'mid', 'high', each a float
            in [0, 1] representing the fraction of total spectral energy in
            that band.  All three sum to 1.0 (or 0.0 if the image is blank).
    """

    # ── Input validation ──
    if image is None:
        return _make_blank_figure(), dict(_ZERO_RATIOS)

    if not isinstance(image, Image.Image):
        try:
            image = Image.open(image)
        except Exception:
            return _make_blank_figure(), dict(_ZERO_RATIOS)

    try:
        # Convert image to grayscale
        grayscale = image.convert("L")
    except Exception:
        return _make_blank_figure(), dict(_ZERO_RATIOS)

    # Resize for consistent analysis
    grayscale = grayscale.resize((224, 224))

    # Convert image to a NumPy array
    image_array = np.asarray(grayscale, dtype=np.float32)

    # Apply 2D FFT
    fft_result = np.fft.fft2(image_array)

    # Shift low frequencies to the center
    fft_shifted = np.fft.fftshift(fft_result)

    # Calculate log magnitude spectrum
    magnitude = np.log1p(np.abs(fft_shifted))

    # Create frequency spectrum visualization
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(magnitude, cmap="inferno")
    axis.set_title("FFT Frequency Spectrum")
    axis.axis("off")
    figure.tight_layout()

    # ---- Frequency band energy ratios ----
    height, width = image_array.shape
    center_y, center_x = height // 2, width // 2
    max_radius = min(height, width) / 2.0

    # Build radial distance map
    y, x = np.ogrid[:height, :width]
    distance = np.sqrt(
        (y - center_y) ** 2 + (x - center_x) ** 2
    )

    # Normalise distance so that 1.0 = half the image dimension
    distance_norm = distance / max_radius

    # Band boundaries (fraction of max_radius)
    LOW_CUTOFF = 0.15   # <= 15% radius
    MID_CUTOFF = 0.50   # 15%–50% radius
    # High = everything beyond 50% radius

    low_mask = distance_norm <= LOW_CUTOFF
    mid_mask = (distance_norm > LOW_CUTOFF) & (distance_norm <= MID_CUTOFF)
    high_mask = distance_norm > MID_CUTOFF

    # Power spectrum (energy)
    power_spectrum = np.abs(fft_shifted) ** 2
    total_energy = power_spectrum.sum()

    # Compute ratios safely
    if total_energy > 0:
        low_ratio = float(power_spectrum[low_mask].sum() / total_energy)
        mid_ratio = float(power_spectrum[mid_mask].sum() / total_energy)
        high_ratio = float(power_spectrum[high_mask].sum() / total_energy)
    else:
        # Blank or constant image — no spectral energy
        low_ratio = 0.0
        mid_ratio = 0.0
        high_ratio = 0.0

    frequency_ratios = {
        "low": low_ratio,
        "mid": mid_ratio,
        "high": high_ratio,
    }

    return figure, frequency_ratios
