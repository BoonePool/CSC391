def edge_directional_map(image, direction_range):
    """
    Generate an edge directional map from a grayscale image.

    Parameters:
    image (numpy.ndarray): Input grayscale image array.
    threshold (float): Threshold value to filter weak edges.

    Returns:
    numpy.ndarray: The edge directional map with angles in degrees.
    """
    from .calculate_gradient import calculate_gradient
    import numpy as np

    gradient_magnitude, gradient_angle = calculate_gradient(image)
    # Apply thresholding to the gradient magnitude
    edge_directional_map = np.where(gradient_angle > direction_range[0] and gradient_angle<direction_range[1], 255, 0).astype(np.float32)
    return edge_directional_map