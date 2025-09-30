import rawpy
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the DNG image
def load_dng_image(file_path):
    raw = rawpy.imread(file_path)
    return raw.raw_image_visible

# Example usage
# Replace with the path to your DNG file
white_image_matrix = load_dng_image("C:\\Users\\Boone Pool\\Downloads\\APC_0001.dng")
black_image_matrix = load_dng_image("C:\\Users\\Boone Pool\\Downloads\\APC_0002.dng")
black_image_matrix2 = load_dng_image("C:\\Users\\Boone Pool\\Downloads\\APC_0015.dng")


white = white_image_matrix[100:200, 100:200].astype(np.float32).flatten()
black = black_image_matrix[100:200, 100:200].astype(np.float32).flatten()
black2 = black_image_matrix2[100:200, 100:200].astype(np.float32).flatten()

# Calculate mean and standard deviation for the selected regions
white_mean = np.mean(white)
white_std = np.std(white)
black_mean = np.mean(black)
black_std = np.std(black)
black2_mean = np.mean(black2)
black2_std = np.std(black2)

# Print the results
print(f"White Image - Mean: {white_mean}, Std: {white_std}, SNR: {white_std/white_mean}")
print(f"Black Image - Mean: {black_mean}, Std: {black_std}, SNR: {black_std/black_mean}")
print(f"Black2 Image - Mean: {black2_mean}, Std: {black2_std}, SNR: {black2_std/black2_mean}")


# Plot histograms for the selected regions

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.hist(white, bins=20, color='blue', alpha=0.7)
plt.title("White Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 3, 2)
plt.hist(black, bins=20, color='red', alpha=0.7)
plt.title("Black Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 3, 3)
plt.hist(black2, bins=20, color='green', alpha=0.7)
plt.title("Black2 Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

