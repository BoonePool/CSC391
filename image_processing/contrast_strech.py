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
