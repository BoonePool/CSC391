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

import matplotlib.pyplot as plt
import cv2
import numpy as np

from .calculate_histogram import equalize_histogram # Load a low-contrast image (grayscale)
image = cv2.imread('image_processing\images\original.jpg', cv2.IMREAD_GRAYSCALE)

# Apply contrast stretching
in_min, in_max = np.percentile(image, (2, 98))  # Use 2nd and 98th percentiles (from copilot suggestion)
stretched_image = contrast_stretch(image, in_min, in_max)

# Apply histogram equalization
equalized_image = equalize_histogram(image)

# Plot the original, stretched, and equalized images and their histograms
fig, axes = plt.subplots(3, 3, figsize=(15, 10))

# Original image and histogram
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')
axes[0, 1].hist(image.ravel(), bins=256, range=(0, 256), color='black')
axes[0, 1].set_title('Original Histogram')

# Stretched image and histogram
axes[1, 0].imshow(stretched_image, cmap='gray')
axes[1, 0].set_title('Contrast Stretched Image')
axes[1, 0].axis('off')
axes[1, 1].hist(stretched_image.ravel(), bins=256, range=(0, 256), color='black')
axes[1, 1].set_title('Stretched Histogram')

# Equalized image and histogram
axes[2, 0].imshow(equalized_image, cmap='gray')
axes[2, 0].set_title('Histogram Equalized Image')
axes[2, 0].axis('off')
axes[2, 1].hist(equalized_image.ravel(), bins=256, range=(0, 256), color='black')
axes[2, 1].set_title('Equalized Histogram')

# Remove unused subplots
axes[0, 2].axis('off')
axes[1, 2].axis('off')
axes[2, 2].axis('off')

plt.tight_layout()
plt.show()