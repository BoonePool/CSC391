def sobel_edge_detector(image, threshold):
    """
    Apply Sobel edge detection to a grayscale image.

    Parameters:
    image (numpy.ndarray): Input grayscale image array.

    Returns:
    numpy.ndarray: The gradient magnitude image after applying Sobel edge detection.
    """
    from .calculate_gradient import calculate_gradient
    import numpy as np

    gradient_magnitude, gradient_angle = calculate_gradient(image)
    # Apply binary thresholding to the gradient magnitude
    binary_image = np.where(gradient_magnitude > threshold, 255, 0).astype(np.uint8)
    return binary_image