import numpy as np
from image_processing.calculate_histogram import calculate_histogram
def equalize_histogram(image, histogram, dist):
    # Compute the cumulative distribution function (CDF)
    cdf = dist.cumsum()
    cdf_normalized = (cdf * 255).astype(np.uint8)  # Normalize to [0, 255]

    # Map the pixel values in the original image to equalized values
    equalized_image = np.interp(image.flatten(), range(256), cdf_normalized).reshape(image.shape).astype(np.uint8)

    return equalized_image