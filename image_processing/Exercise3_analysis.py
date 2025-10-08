def sobel_edge_detector(image, threshold):
    """
    Apply Sobel edge detection to a grayscale image.

    Parameters:
    image (numpy.ndarray): Input grayscale image array.

    Returns:
    numpy.ndarray: The gradient magnitude image after applying Sobel edge detection.
    """
    import numpy as np

    gradient_magnitude, gradient_angle = calculate_gradient(image)
    # Apply binary thresholding to the gradient magnitude
    binary_image = np.where(gradient_magnitude > threshold, 255, 0).astype(np.uint8)
    return binary_image
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
    gradient_angle = np.arctan2(grad_y, grad_x)
    return gradient_magnitude,  gradient_angle

def edge_directional_map(image, direction_range):
    """
    Generate an edge directional map from a grayscale image.

    Parameters:
    image (numpy.ndarray): Input grayscale image array.
    threshold (float): Threshold value to filter weak edges.

    Returns:
    numpy.ndarray: The edge directional map with angles in degrees.
    """
    import numpy as np


    gradient_magnitude, gradient_angle = calculate_gradient(image)
    # Apply thresholding to the gradient magnitude
    edge_directional_map = np.where((gradient_angle > direction_range[0]) & (gradient_angle < direction_range[1]), 255, 0).astype(np.float32)
    return edge_directional_map


import cv2
import matplotlib.pyplot as plt

# Load the image as grayscale
image_path = 'C:\\Users\\Boone Pool\\CSC391\\image_processing\\images\\kitten.jpg'  # Replace with the path to your image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Sobel edge detector
threshold = 50  # Adjust threshold as needed
sobel_edges = sobel_edge_detector(image, threshold)

# Apply directional edge detector for edges at roughly 45 degrees
direction_range = (np.pi / 4 - .05, np.pi / 4 + .05) 
directional_edges = edge_directional_map(image, direction_range)

canny_edges = cv2.Canny(image, 100, 200) 

# Display the results
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.title("Sobel Edge Detector")
plt.imshow(sobel_edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Directional Edge Detector")
plt.imshow(directional_edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Canny Edge Detector")
plt.imshow(canny_edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()