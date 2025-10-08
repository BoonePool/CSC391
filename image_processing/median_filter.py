def median_filter(image, kernel_size=3):
    """
    Apply a median filter to the input image.

    Parameters:
    image (numpy.ndarray): Input image array.
    kernel_size (int): Size of the median filter kernel. Must be an odd integer.

    Returns:
    numpy.ndarray: The filtered image.
    """
    import numpy as np
    filtered_image = np.zeros_like(image)
    kernel_elements = []
    
    for i in range(kernel_size//2, image.shape[0]):
        for j in range(kernel_size//2, image.shape[1]):
            for u in range(-kernel_size//2, kernel_size//2 + 1):
                for k in range(1, kernel_size//2 + 1):
                    kernel_elements.append(image[i+u,j+k])
            filtered_image[i, j] = np.median(kernel_elements)
    
    return filtered_image
