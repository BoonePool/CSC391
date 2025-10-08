def contrast_stretch(image, in_min, in_max):
    """
    Perform contrast stretching on the input image.
    
    Parameters:
    - image: Input image as a NumPy array.
    - in_min: Minimum pixel value in the input image to be stretched.
    - in_max: Maximum pixel value in the input image to be stretched.
    - out_min: Minimum pixel value in the output image after stretching.
    - out_max: Maximum pixel value in the output image after stretching.

    Returns:
    - Stretched image as a NumPy array.
    """
    out_min = 0
    out_max = 255
    import numpy as np

    # Ensure the input values are within valid range
    if in_min >= in_max:
        raise ValueError("Invalid min/max values for contrast stretching.")

    # Perform contrast stretching
    stretched = ((image - in_min)/(in_max - in_min)) * (out_max - out_min) + out_min
    stretched = np.clip(stretched, out_min, out_max)  # Clip values to the output range

    return stretched.astype(image.dtype)


import numpy as np
def equalize_histogram(image, dist):
    # Compute the cumulative distribution function (CDF)
    cdf = dist.cumsum()
    cdf_normalized = (cdf * 255).astype(np.uint8)  # Normalize to [0, 255]

    # Map the pixel values in the original image to equalized values
    equalized_image = np.interp(image.flatten(), range(256), cdf_normalized).reshape(image.shape).astype(np.uint8)

    return equalized_image

import numpy as np

def calculate_histogram(image, bins):
    # Initialize histogram with the number of bins
    histogram = np.zeros(bins, dtype=int)
    range = 256 // bins
    # Calculate the histogram
    for value in image.flatten():
        histogram[value//range] += 1
    dist = histogram / image.size
    return histogram, dist


import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# Load a low-contrast image from an example image path
image_path = "C:\\Users\\Boone Pool\\CSC391\\image_processing\\images\\kitten.jpg"
low_contrast_image = io.imread(image_path, as_gray=True)
low_contrast_image = (low_contrast_image * 255).astype(np.uint8)  # Convert to 8-bit grayscale

# Apply contrast stretching
stretched_image = contrast_stretch(low_contrast_image, in_min=low_contrast_image.min(), in_max=low_contrast_image.max())

# Calculate histograms for the original and stretched images
original_hist, original_dist = calculate_histogram(low_contrast_image, bins=256)
stretched_hist, stretched_dist = calculate_histogram(stretched_image, bins=256)

# Apply histogram equalization
equalized_image = equalize_histogram(low_contrast_image, original_dist)

# Calculate histogram for the equalized image
equalized_hist, _ = calculate_histogram(equalized_image, bins=256)

# Plot the original, stretched, and equalized histograms
fig, axs = plt.subplots(3, 2, figsize=(12, 8))

# Original image and histogram
axs[0, 0].imshow(low_contrast_image, cmap='gray')
axs[0, 0].set_title("Original Image")
axs[0, 0].axis('off')
axs[0, 1].bar(range(256), original_hist, color='gray')
axs[0, 1].set_title("Original Histogram")

# Stretched image and histogram
axs[1, 0].imshow(stretched_image, cmap='gray')
axs[1, 0].set_title("Stretched Image")
axs[1, 0].axis('off')
axs[1, 1].bar(range(256), stretched_hist, color='gray')
axs[1, 1].set_title("Stretched Histogram")

# Equalized image and histogram
axs[2, 0].imshow(equalized_image, cmap='gray')
axs[2, 0].set_title("Equalized Image")
axs[2, 0].axis('off')
axs[2, 1].bar(range(256), equalized_hist, color='gray')
axs[2, 1].set_title("Equalized Histogram")

plt.tight_layout()
plt.show()