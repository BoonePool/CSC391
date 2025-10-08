import numpy as np

def apply_convolution(image, kernel):
    """
    Apply convolution to a single-channel image with a given kernel.
    No use of cv2.filter2D.
    """
    # Ensure kernel is numpy array
    kernel = np.array(kernel)
    k_size = kernel.shape[0]
    pad = k_size // 2

    # Pad the image (zero padding chosen here)
    padded = np.pad(image, pad, mode='constant', constant_values=0)

    # Output image
    output = np.zeros_like(image, dtype=np.float32)

    # Convolution loop
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+k_size, j:j+k_size]
            output[i, j] = np.sum(region * kernel)

    # Clip to valid range and return uint8
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def calculate_gradient(image):
    """
    Calculate the gradient magnitude of the input image using Sobel filters.

    Parameters:
    image (numpy.ndarray): Input grayscale image array.

    Returns:
    numpy.ndarray: The gradient magnitude image.
    """
    import numpy as np

    # Define Sobel filters
    sobel_filter_vert = np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]], dtype=np.float32)
    sobel_filter_horz = np.array([[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]], dtype=np.float32)

    # Apply Sobel filters to get gradients in x and y directions
    grad_x = apply_convolution(image, sobel_filter_vert)
    grad_y = apply_convolution(image, sobel_filter_horz)

    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_x.astype(np.float32)**2 + grad_y.astype(np.float32)**2)

    # Clip to valid range and return uint8
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

    return gradient_magnitude

