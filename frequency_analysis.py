import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def analyze_frequency(image):
    """
    Analyze the frequency components of an image
    using the Fast Fourier Transform (FFT).
    """

    # Convert image to grayscale
    grayscale = image.convert("L")

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

    # Calculate a basic high-frequency energy ratio
    height, width = image_array.shape
    y, x = np.ogrid[:height, :width]

    center_y, center_x = height // 2, width // 2
    radius = min(height, width) * 0.15

    distance = np.sqrt(
        (y - center_y) ** 2 + (x - center_x) ** 2
    )

    high_frequency_mask = distance >= radius

    power_spectrum = np.abs(fft_shifted) ** 2
    total_energy = power_spectrum.sum()

    if total_energy > 0:
        high_frequency_ratio = (
            power_spectrum[high_frequency_mask].sum()
            / total_energy
        )
    else:
        high_frequency_ratio = 0.0

    return figure, float(high_frequency_ratio)
