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